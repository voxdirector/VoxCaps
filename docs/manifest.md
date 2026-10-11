# Input and output

Manifests contain one JSON object per line. Every record needs a unique `id`. Audio paths are relative to the input manifest; stage outputs use resolved paths so they remain usable across directories. Source audio must be PCM-WAV.

## Source records

Collected audio:

```json
{"id":"scene_001","audio":"media/scene.wav","media_type":"short_drama","languages":["zh"]}
```

Synthesis job:

```json
{
  "id":"pronunciation_001",
  "source_type":"targeted_synthesis",
  "synthetic_subset":"difficult_pronunciation",
  "student_text":"给予好评。",
  "reference_audio":"references/speaker.wav",
  "reference_transcript":"您好，欢迎来到这里。",
  "languages":["zh"],
  "pronunciation_indices":[0],
  "reading_overrides":{"0":"jǐ"}
}
```

Synthetic subsets are `elderly_timbres`, `short_utterances` and `difficult_pronunciation`. Elderly references need `reference_age_group: elderly`. Reading overrides use character offsets and tone-mark pinyin. The generated manifest retains the original `student_text`.

## Preprocessing and annotation

Stage 2 writes `audio` for the original mixture, `analysis_audio` for vocals, `transcript` and `turns`. Word times are relative to the clip:

```json
{"speaker_id":"S1","text":"你好世界","language":"zh","end_time":1.9,"words":[
  {"text":"你好","start_time":0.1,"end_time":0.6},
  {"text":"世界","start_time":0.9,"end_time":1.4}
]}
```

Stage 3 adds `annotation` with `environment`, `speakers` and `content`. Speaker entries may use `{id, description}` objects or `Subject1: ...` strings. Speech uses `<S1>...</S1>` tags; sound events use `<Audio>...</Audio>`. IDs follow first appearance. Refinement also accepts old `env`, `bgm`, `speakers`, `local` annotation caches.

## Refinement

Stage 4 computes quality and attribute results when model settings are supplied. Cached measurements may instead be provided in `quality_score` with numeric `pesq`, `stoi`, `si_sdr` and `mos`. DNSMOS measurements use `dnsmos`.

Cached attributes use `attribute_annotations`:

```json
{"speakers":[{"id":"S1","gender":{"value":"女性","evidence":"青年女性"},"age":{"value":"青年","evidence":"青年女性"},"persona":null}],
 "turns":[{"turn_index":0,"speaker_id":"S1","emotion":{"value":"温柔","evidence":"温柔地说"}}]}
```

Each attribute is `null` or has a value and a verbatim evidence span from the corresponding speaker or turn description. Word alignment and tagged speech must preserve the transcript's words.

Accepted records contain `waveform_target`, `annotation_audio`, `analysis_audio`, `description`, `attributes`, scores, raw annotation and correction history. Rejected records contain the ID and rejection reasons. The report records counts and the data root. A `fixture_only` flag identifies technical examples.

## Sorting

Score sorting reads `id` and a numeric field such as `score` or `quality_score.mos`. Preference sorting reads pairs:

```json
{"preferred":"a","dispreferred":"b"}
```

Preference output groups candidates into topological frontiers. Candidates in one frontier have no assigned order, and candidates in different frontiers may still be incomparable.
