# Google OAuth Setup Steps

## 🔐 Complete Google OAuth Flow (Already Built!)

### Step-by-Step Process:

```
User visits landing page
        ↓
Clicks "Start Indexing with Google"
        ↓
[Redirected to Google Sign-in]
        ↓
User signs into their Google account
        ↓
Google shows permission consent screen:
  ✓ Read your email & profile
  ✓ View files in your Google Drive (read-only)
        ↓
User clicks "Allow"
        ↓
[Google redirects back to your app]
        ↓
Backend receives authorization code
        ↓
Backend exchanges code for access tokens
        ↓
Backend creates/updates user in database
        ↓
Backend stores Google tokens securely
        ↓
Backend sets authentication cookie
        ↓
User redirected to dashboard
        ↓
✅ User is now authenticated!
```

---

## 📋 What's Already Implemented

### 1. **OAuth Client Configuration** ✅

```python
# backend/app/auth/google.py (lines 10-20)
google_oauth_client = GoogleOAuth2(
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    scopes=[
        "openid",           # Basic user info
        "email",            # User's email
        "profile",          # User's profile
        "https://www.googleapis.com/auth/drive.readonly",  # Drive read-only
    ],
)
```

### 2. **OAuth Routes** ✅

```python
# backend/app/routers/auth.py (lines 24-36)
# Automatically creates these endpoints:
# GET  /auth/google/authorize     - Starts OAuth flow
# GET  /auth/google/callback      - Handles Google's redirect
```

### 3. **Landing Page Button** ✅

```html
<!-- backend/app/templates/landing.html -->
<a href="/auth/google/authorize" class="btn btn-primary">
    Start Indexing with Google
</a>
```

### 4. **Token Storage** ✅

```python
# backend/app/auth/users.py (lines 89-108)
# Automatically stores in database:
- google_access_token      # For API calls
- google_refresh_token     # To renew access
- google_token_expires_at  # Expiration time
```

### 5. **User Database Model** ✅

```python
# backend/app/models/user.py
# Already has fields for:
- google_access_token
- google_refresh_token  
- google_token_expires_at
```

---

## 🎯 What You Need to Do (Setup Only)

**No development needed!** Just configuration:

### 1. Create Google OAuth App

Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials):

1. **Create Project** (if you don't have one)
2. **Enable APIs:**
   - Google Drive API
   - Google Picker API

3. **Create OAuth 2.0 Client ID:**
   - Type: **Web application**
   - Authorized redirect URI:
     ```
     http://localhost:8000/auth/google/callback
     ```
   - Copy **Client ID** and **Client Secret**

4. **Create API Key:**
   - For Google Picker
   - Copy the key

### 2. Add Credentials to `.env`

```env
GOOGLE_CLIENT_ID=your-id-here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret-here
GOOGLE_API_KEY=your-api-key-here
```

### 3. Test the Flow

1. Start server (already running at `http://localhost:8000`)
2. Click "Start Indexing with Google"
3. Sign in with Google
4. Grant permissions
5. ✅ Done! You're authenticated

---

## 🔒 Security Features (Already Built In)

1. **Read-only Drive access** - Can't modify user files
2. **Secure token storage** - Tokens encrypted in database
3. **JWT cookies** - HttpOnly, secure session management
4. **Token refresh** - Automatically renews expired tokens
5. **Associate by email** - Links OAuth accounts to existing users

---

## 🎨 User Experience Flow

### First-time User:
```
Landing Page → Sign in with Google → Grant permissions → Dashboard
```

### Returning User:
```
Landing Page → Already logged in → Dashboard
```

---

## 🔧 No Additional Development Needed!

Everything is already implemented. The only missing piece is **your Google OAuth credentials** from Google Cloud Console.

Once you add those three values to `.env`:
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_API_KEY`

The authentication will work perfectly out of the box!

---

## 🧪 Testing the Authentication

### Test 1: Check Landing Page
1. Open browser: http://localhost:8000
2. You should see the landing page with "Start Indexing with Google" button

### Test 2: Start OAuth Flow
1. Click **"Start Indexing with Google"**
2. You should be redirected to Google's sign-in page
3. Sign in with your Google account
4. Grant permissions (Drive read-only)
5. You should be redirected back to the dashboard

### Test 3: Verify Authentication
After successful OAuth:
- Check: http://localhost:8000/auth/me
- You should see your user details as JSON

### Test 4: Check Dashboard
- Go to: http://localhost:8000/dashboard
- You should see:
  - Your email displayed
  - Stats cards (clips indexed, searches)
  - "Choose Folder from Drive" button

### Test 5: Test Google Picker
1. On dashboard, click **"Choose Folder from Drive"**
2. Google Picker modal should open
3. Select a folder with videos
4. File list should appear with validation badges (✅/❌)

---

## 🐛 Common Issues

### Issue 1: "Redirect URI mismatch"
**Fix:** Make sure in Google Cloud Console, you added:
```
http://localhost:8000/auth/google/callback
```

### Issue 2: "Google OAuth credentials not set"
**Fix:** Check your `.env` has all three values:
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`  
- `GOOGLE_API_KEY`

### Issue 3: "Picker doesn't open"
**Fix:** 
- Enable **Google Picker API** in Google Cloud Console
- Make sure `GOOGLE_API_KEY` is set in `.env`

---

## 🔍 Debug Endpoints

Check these URLs to verify setup:

- **Health check:** http://localhost:8000/health
- **API docs:** http://localhost:8000/docs
- **Current user:** http://localhost:8000/auth/me (after sign-in)
- **Token status:** http://localhost:8000/api/drive/user/token-status (after sign-in)

