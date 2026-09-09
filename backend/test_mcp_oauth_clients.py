"""Unit tests for MCP OAuth DCR, CIMD validation, and redirect/SSRF policy."""

import ipaddress
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.mcp_server.clients import (
    CimdError,
    DcrError,
    _is_blocked_ip,
    cimd_url_precheck,
    decode_dcr_client,
    hash_client_secret,
    is_allowed_redirect_uri,
    is_cimd_client_id,
    issue_dcr_client,
    parse_cimd_document,
    resolve_oauth_client,
    verify_client_secret,
)


class RedirectUriPolicyTests(unittest.TestCase):
    def test_https_is_allowed(self):
        self.assertTrue(
            is_allowed_redirect_uri("https://chatgpt.com/connector/oauth/callback")
        )

    def test_localhost_http_with_port_is_allowed(self):
        self.assertTrue(
            is_allowed_redirect_uri("http://localhost:6274/oauth/callback")
        )
        self.assertTrue(
            is_allowed_redirect_uri("http://127.0.0.1:6274/oauth/callback")
        )

    def test_http_non_loopback_rejected(self):
        self.assertFalse(is_allowed_redirect_uri("http://example.com/callback"))

    def test_fragment_rejected(self):
        self.assertFalse(
            is_allowed_redirect_uri("https://example.com/callback#frag")
        )

    def test_javascript_and_relative_rejected(self):
        self.assertFalse(is_allowed_redirect_uri("javascript:alert(1)"))
        self.assertFalse(is_allowed_redirect_uri("/callback"))
        self.assertFalse(is_allowed_redirect_uri(""))


class DcrRoundTripTests(unittest.TestCase):
    def test_public_client_jwt_round_trip(self):
        registered = issue_dcr_client(
            {
                "redirect_uris": ["http://127.0.0.1:6274/callback"],
                "client_name": "MCP Inspector",
                "token_endpoint_auth_method": "none",
            }
        )
        self.assertIn("client_id", registered)
        self.assertEqual(registered["token_endpoint_auth_method"], "none")
        self.assertNotIn("client_secret", registered)
        self.assertEqual(
            registered["redirect_uris"], ["http://127.0.0.1:6274/callback"]
        )

        client = decode_dcr_client(registered["client_id"])
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.kind, "dcr")
        self.assertEqual(client.redirect_uris, ("http://127.0.0.1:6274/callback",))
        self.assertEqual(client.client_name, "MCP Inspector")
        self.assertFalse(client.confidential)

        resolved = resolve_oauth_client(registered["client_id"])
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "dcr")
        self.assertIn(
            "http://127.0.0.1:6274/callback", resolved.redirect_uris
        )
        self.assertNotIn(
            "https://evil.example/callback", resolved.redirect_uris
        )

    def test_confidential_client_secret_hashed_in_jwt(self):
        registered = issue_dcr_client(
            {
                "redirect_uris": ["https://app.example.com/oauth/callback"],
                "token_endpoint_auth_method": "client_secret_post",
            }
        )
        secret = registered["client_secret"]
        client = decode_dcr_client(registered["client_id"])
        self.assertIsNotNone(client)
        assert client is not None
        self.assertTrue(client.confidential)
        self.assertEqual(
            client.client_secret_hash, hash_client_secret(secret)
        )
        self.assertTrue(verify_client_secret(client, secret))
        self.assertFalse(verify_client_secret(client, "wrong-secret"))

    def test_invalid_redirect_rejected(self):
        with self.assertRaises(DcrError) as ctx:
            issue_dcr_client({"redirect_uris": ["http://evil.example/callback"]})
        self.assertEqual(ctx.exception.error, "invalid_redirect_uri")

    def test_missing_redirect_uris_rejected(self):
        with self.assertRaises(DcrError) as ctx:
            issue_dcr_client({"client_name": "Nope"})
        self.assertEqual(ctx.exception.error, "invalid_redirect_uri")


class CimdDocumentTests(unittest.TestCase):
    URL = "https://app.example.com/oauth/client-metadata.json"

    def _doc(self, **overrides):
        doc = {
            "client_id": self.URL,
            "client_name": "Example MCP Client",
            "redirect_uris": ["http://127.0.0.1:3000/callback"],
            "token_endpoint_auth_method": "none",
        }
        doc.update(overrides)
        return doc

    def test_valid_document(self):
        client = parse_cimd_document(self.URL, self._doc())
        self.assertEqual(client.kind, "cimd")
        self.assertEqual(client.client_name, "Example MCP Client")
        self.assertEqual(
            client.redirect_uris, ("http://127.0.0.1:3000/callback",)
        )

    def test_client_id_mismatch_rejected(self):
        with self.assertRaises(CimdError):
            parse_cimd_document(
                self.URL,
                self._doc(client_id="https://other.example/client.json"),
            )

    def test_missing_name_rejected(self):
        with self.assertRaises(CimdError):
            parse_cimd_document(self.URL, self._doc(client_name=""))

    def test_invalid_redirect_in_document_rejected(self):
        with self.assertRaises(CimdError):
            parse_cimd_document(
                self.URL,
                self._doc(redirect_uris=["http://evil.example/callback"]),
            )

    def test_https_url_resolves_as_cimd_stub(self):
        self.assertTrue(is_cimd_client_id(self.URL))
        stub = resolve_oauth_client(self.URL)
        self.assertIsNotNone(stub)
        assert stub is not None
        self.assertEqual(stub.kind, "cimd")
        self.assertEqual(stub.redirect_uris, ())


class SsrfUrlTests(unittest.TestCase):
    def test_loopback_and_private_ips_blocked(self):
        self.assertIsNotNone(cimd_url_precheck("https://127.0.0.1/client.json"))
        self.assertIsNotNone(cimd_url_precheck("https://localhost/client.json"))
        self.assertIsNotNone(cimd_url_precheck("https://192.168.1.10/client.json"))
        self.assertIsNotNone(cimd_url_precheck("https://10.0.0.1/client.json"))
        self.assertIsNotNone(cimd_url_precheck("https://169.254.169.254/latest"))

    def test_http_and_root_path_blocked(self):
        self.assertIsNotNone(cimd_url_precheck("http://example.com/client.json"))
        self.assertIsNotNone(cimd_url_precheck("https://example.com/"))
        self.assertIsNotNone(cimd_url_precheck("https://example.com"))

    def test_public_https_with_path_passes_precheck(self):
        self.assertIsNone(
            cimd_url_precheck("https://app.example.com/oauth/client.json")
        )

    def test_blocked_ip_helper(self):
        self.assertTrue(_is_blocked_ip(ipaddress.ip_address("127.0.0.1")))
        self.assertTrue(_is_blocked_ip(ipaddress.ip_address("10.1.2.3")))
        self.assertTrue(_is_blocked_ip(ipaddress.ip_address("::1")))
        self.assertFalse(_is_blocked_ip(ipaddress.ip_address("1.1.1.1")))


if __name__ == "__main__":
    unittest.main()
