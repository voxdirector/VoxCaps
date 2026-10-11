# Paper settings

| Stage | Setting | Code or configuration |
| --- | --- | --- |
| Coverage Design | Chinese and English media; three synthetic subsets of 100,000 examples each | [Coverage profile](../configs/01_coverage_design/paper_profile.json), [source accounting](../src/voxcaps/coverage_design/manifest.py) |
| Coverage Design | CosyVoice 3 teacher; pypinyin hints only for synthesis; original student spelling retained | [Synthesis](../src/voxcaps/coverage_design/synthesis.py) |
| Speech Preprocessing | UVR vocals for analysis; original mixture for annotation and the waveform target | [UVR](../src/voxcaps/speech_preprocessing/uvr/api.py), [clip preparation](../src/voxcaps/speech_preprocessing/segmentation.py) |
| Speech Preprocessing | 3D-Speaker, VAD and CAM++; single-speaker clips of 1 to 60 seconds; multi-speaker clips up to 120 seconds | [Diarization](../src/voxcaps/speech_preprocessing/diarization.py), [clip assembly](../src/voxcaps/speech_preprocessing/segmentation.py) |
| Speech Preprocessing | Fun-ASR-Nano, SenseVoice and Qwen3-ForcedAligner | [Model calls](../src/voxcaps/speech_preprocessing/models.py) |
| Scope-Structured Annotation | Qwen3.5-Omni; Environment, Speakers and Content; media-specific persona guidance | [Annotator](../src/voxcaps/scope_annotation/annotator.py), [style matrix](../configs/03_scope_annotation/style_matrix.json) |
| Data Refinement | DNSMOS/SQUIM measurements; minimum PESQ estimate 2.0, STOI 0.85, SI-SDR 0 and MOS 2.5 | [Scorers](../src/voxcaps/data_refinement/), [filter settings](../configs/04_data_refinement/filter_defaults.json) |
| Data Refinement | Short synthetic utterances exempt from the lower duration limit | [Coverage metadata](../src/voxcaps/coverage_design/manifest.py), [quality gates](../src/voxcaps/data_refinement/quality.py) |
| Data Refinement | Tag repair, pause-based punctuation and description attributes | [Tag repair](../src/voxcaps/data_refinement/tag_repair.py), [punctuation](../src/voxcaps/data_refinement/description.py), [attributes](../src/voxcaps/data_refinement/attributes.py) |

Pause thresholds of 0.08, 0.18 and 0.45 seconds come from the existing punctuation code. The paper describes the procedure without specifying these values. Manual review and expressive-subset selection are omitted; sorting existing results is an optional utility.
