-- Indexing RPCs for the PostgREST worker path.
--
-- The Service Bus triggers no longer hold Postgres connections; they call these
-- functions over PostgREST. Two things cannot be expressed as a plain PATCH and
-- so live here:
--
--   1. Progress counters. frames_completed/frames_failed are incremented by up
--      to FRAME_INDEX_PARALLELISM frames at once, and each increment has to be
--      followed by a "is the video finished" check. Read-modify-write over HTTP
--      would lose counts and could leave a video stuck in 'processing'.
--   2. vector columns. PostgREST cannot infer that a JSON array is destined for
--      a pgvector column, so embeddings arrive as text and are cast here.
--
-- Apply with `supabase db push`, or paste into the SQL editor. Afterwards reload
-- the API schema cache:  notify pgrst, 'reload schema';

set check_function_bodies = off;

-- ---------------------------------------------------------------------------
-- Shared completion check
-- ---------------------------------------------------------------------------

create or replace function public.distill_finalize_video_indexing(p_video_id uuid)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  v_row public.indexed_files%rowtype;
  v_now timestamp := (now() at time zone 'utc');
begin
  -- Callers already hold FOR NO KEY UPDATE on this row; re-taking the same mode
  -- is a no-op for them and still serialises a direct call.
  select * into v_row
    from public.indexed_files
   where id = p_video_id
   for no key update;

  if not found or v_row.frames_total is null then
    return null;
  end if;

  if v_row.frames_completed + v_row.frames_failed < v_row.frames_total then
    return v_row.indexing_status;
  end if;

  if v_row.frames_failed >= v_row.frames_total then
    update public.indexed_files
       set indexing_status = 'failed',
           error_message = 'All frames failed to index',
           updated_at = v_now
     where id = p_video_id;
    return 'failed';
  end if;

  update public.indexed_files
     set indexing_status = 'completed',
         error_message = null,
         indexed_at = v_now,
         updated_at = v_now
   where id = p_video_id;
  return 'completed';
end;
$$;

-- ---------------------------------------------------------------------------
-- Frame embedding upsert + progress, in one transaction
-- ---------------------------------------------------------------------------

create or replace function public.distill_upsert_frame_embedding(
  p_video_id uuid,
  p_frame_index integer,
  p_time_seconds double precision,
  p_embedding text,
  p_frame_image_url text default null,
  p_count_completion boolean default true
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_inserted boolean;
  v_status text;
begin
  -- Lock the parent row BEFORE touching the child table. Inserting into
  -- video_frame_embeddings takes a FOR KEY SHARE lock on this indexed_files row
  -- via the foreign key; incrementing the counter afterwards would upgrade that
  -- lock, and concurrent frames upgrading at once deadlock. Taking the stronger
  -- lock up front makes every frame queue in a single order instead.
  if p_count_completion then
    perform 1 from public.indexed_files where id = p_video_id for no key update;
  end if;

  insert into public.video_frame_embeddings
      (id, video_id, frame_index, time_seconds, embedding, frame_image_url)
  values
      (gen_random_uuid(), p_video_id, p_frame_index, p_time_seconds,
       p_embedding::vector, p_frame_image_url)
  on conflict on constraint uq_video_frame_video_id_frame_index
  do update
     set time_seconds = excluded.time_seconds,
         embedding = excluded.embedding,
         frame_image_url = excluded.frame_image_url
  returning (xmax = 0) into v_inserted;

  -- Only a genuinely new row advances progress; re-indexing an existing frame
  -- must not inflate frames_completed past frames_total.
  if p_count_completion and v_inserted then
    update public.indexed_files
       set frames_completed = frames_completed + 1,
           updated_at = (now() at time zone 'utc')
     where id = p_video_id;
    v_status := public.distill_finalize_video_indexing(p_video_id);
  end if;

  return jsonb_build_object('inserted', v_inserted, 'indexing_status', v_status);
end;
$$;

-- ---------------------------------------------------------------------------
-- Frame failure accounting
-- ---------------------------------------------------------------------------

create or replace function public.distill_record_frame_failure(
  p_video_id uuid,
  p_frame_index integer default null
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_status text;
begin
  -- Same lock ordering as distill_upsert_frame_embedding: parent row first.
  perform 1 from public.indexed_files where id = p_video_id for no key update;

  -- A frame that already stored an embedding is not a failure: a later retry of
  -- the same frame would otherwise be counted twice against frames_total.
  if p_frame_index is not null and exists (
    select 1
      from public.video_frame_embeddings
     where video_id = p_video_id
       and frame_index = p_frame_index
  ) then
    return jsonb_build_object('counted', false);
  end if;

  update public.indexed_files
     set frames_failed = frames_failed + 1,
         updated_at = (now() at time zone 'utc')
   where id = p_video_id;

  v_status := public.distill_finalize_video_indexing(p_video_id);
  return jsonb_build_object('counted', true, 'indexing_status', v_status);
end;
$$;

-- ---------------------------------------------------------------------------
-- vector-column writes on indexed_files
-- ---------------------------------------------------------------------------

create or replace function public.distill_set_thumbnail_embedding(
  p_file_id uuid,
  p_embedding text
)
returns void
language sql
security definer
set search_path = public
as $$
  update public.indexed_files
     set thumbnail_embedding = p_embedding::vector,
         updated_at = (now() at time zone 'utc')
   where id = p_file_id;
$$;

create or replace function public.distill_set_vision_embedding(
  p_file_id uuid,
  p_embedding text,
  p_status text default 'completed',
  p_indexed_at timestamp default null
)
returns void
language sql
security definer
set search_path = public
as $$
  update public.indexed_files
     set vision_embedding = p_embedding::vector,
         vision_indexing_status = p_status,
         vision_indexed_at = coalesce(p_indexed_at, (now() at time zone 'utc')),
         error_message = null,
         updated_at = (now() at time zone 'utc')
   where id = p_file_id;
$$;

create or replace function public.distill_set_color_signature(
  p_file_id uuid,
  p_histogram text,
  p_palette jsonb,
  p_mean_l double precision,
  p_std_l double precision,
  p_mean_a double precision,
  p_mean_b double precision
)
returns void
language sql
security definer
set search_path = public
as $$
  update public.indexed_files
     set color_histogram = p_histogram::vector,
         color_palette = p_palette,
         color_mean_l = p_mean_l,
         color_std_l = p_std_l,
         color_mean_a = p_mean_a,
         color_mean_b = p_mean_b,
         updated_at = (now() at time zone 'utc')
   where id = p_file_id;
$$;

-- ---------------------------------------------------------------------------
-- Transcript segment swap
-- ---------------------------------------------------------------------------

create or replace function public.distill_replace_transcript_segments(
  p_video_id uuid,
  p_segments jsonb
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  v_count integer;
begin
  -- Delete and insert in one transaction so a re-index never leaves the video
  -- with a partial transcript.
  delete from public.video_transcript_segments where video_id = p_video_id;

  insert into public.video_transcript_segments
      (id, video_id, segment_index, start_seconds, end_seconds, text, words, text_embedding)
  select
      gen_random_uuid(),
      p_video_id,
      (seg->>'segment_index')::integer,
      (seg->>'start_seconds')::double precision,
      (seg->>'end_seconds')::double precision,
      seg->>'text',
      case when seg->'words' = 'null'::jsonb then null else seg->'words' end,
      case when seg->>'text_embedding' is null then null
           else (seg->>'text_embedding')::vector
      end
    from jsonb_array_elements(coalesce(p_segments, '[]'::jsonb)) as seg;

  get diagnostics v_count = row_count;
  return v_count;
end;
$$;

-- ---------------------------------------------------------------------------
-- Expose to the PostgREST roles the workers authenticate as
-- ---------------------------------------------------------------------------

grant execute on function public.distill_finalize_video_indexing(uuid) to service_role;
grant execute on function public.distill_upsert_frame_embedding(uuid, integer, double precision, text, text, boolean) to service_role;
grant execute on function public.distill_record_frame_failure(uuid, integer) to service_role;
grant execute on function public.distill_set_thumbnail_embedding(uuid, text) to service_role;
grant execute on function public.distill_set_vision_embedding(uuid, text, text, timestamp) to service_role;
grant execute on function public.distill_set_color_signature(uuid, text, jsonb, double precision, double precision, double precision, double precision) to service_role;
grant execute on function public.distill_replace_transcript_segments(uuid, jsonb) to service_role;

notify pgrst, 'reload schema';
