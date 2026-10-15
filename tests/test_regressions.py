"""Regression checks using real audio conversion and no model downloads."""

import tempfile
import unittest
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from omnivoice import VoiceClonePrompt

from voice_clone_webui import app, audio_downloader, profile_manager, transcriber, voice_cloner


class UnsafePayload:
    def __reduce__(self):
        return eval, ("1 + 1",)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        for module in (app, audio_downloader, transcriber):
            self.enterContext(patch.object(module, "TMP_DIR", self.root))
        for module in (app, profile_manager):
            self.enterContext(patch.object(module, "VOICE_PROFILES_DIR", self.root))

    def audio(self, name="source.wav", *, rate=16000, channels=1, subtype="PCM_16"):
        path = self.root / name
        shape = (rate // 10,) if channels == 1 else (rate // 10, channels)
        sf.write(path, np.full(shape, 0.1), rate, subtype=subtype)
        return path

    def prompt(self):
        return VoiceClonePrompt(
            ref_audio_tokens=torch.tensor([[1, 2]], dtype=torch.int64),
            ref_text="reference words",
            ref_rms=0.1,
        )

    def test_profile_round_trip_preserves_prompt_audio_and_transcript(self):
        source = self.audio()
        saved = profile_manager.save_profile("speaker", self.prompt(), "words", source)
        raw = torch.load(saved, weights_only=True)
        self.assertIsInstance(raw["ref_audio_wav"], torch.Tensor)
        self.assertEqual(raw["engine"], "omnivoice")
        self.assertEqual(raw["format_version"], 2)
        loaded = profile_manager.load_profile("speaker")
        self.assertEqual(loaded["ref_text"], "words")
        item = loaded["voice_clone_prompt"]
        self.assertIsInstance(item, VoiceClonePrompt)
        self.assertEqual(item.ref_text, "reference words")
        self.assertTrue(torch.equal(item.ref_audio_tokens, self.prompt().ref_audio_tokens))
        self.assertEqual(item.ref_rms, 0.1)
        np.testing.assert_allclose(loaded["ref_audio_wav"], sf.read(source)[0])

    def test_profile_loader_rejects_executable_pickle(self):
        torch.save(UnsafePayload(), self.root / "unsafe.pt")
        with self.assertRaisesRegex(ValueError, "legacy pickle"):
            profile_manager.load_profile("unsafe")

    def test_legacy_profiles_are_rejected_with_recovery_guidance(self):
        torch.save({"payload": UnsafePayload()}, self.root / "legacy.pt")
        with self.assertRaisesRegex(ValueError, "Recreate"):
            profile_manager.load_profile("legacy")
        self.assertTrue((self.root / "legacy.pt").exists())

    def test_previous_engine_profile_reuses_audio_without_loading_old_prompt(self):
        path = self.root / "previous.pt"
        torch.save({
            "format_version": 1,
            "name": "previous",
            "ref_text": "reference words",
            "ref_audio_wav": torch.full((1600,), 0.1),
            "ref_audio_sr": 16000,
            "voice_clone_prompt": [{"ref_code": torch.tensor([[1, 2]])}],
        }, path)
        original = path.read_bytes()
        profile = profile_manager.load_profile("previous")
        self.assertIsNone(profile["voice_clone_prompt"])
        self.assertTrue(profile["needs_prompt_regeneration"])
        status, text, audio, cached, reference = app.on_load_profile("previous")
        self.assertIn("save again", status)
        self.assertEqual(text, "reference words")
        self.assertIsNone(cached)
        self.assertEqual(audio, reference)
        self.assertEqual(sf.info(audio).frames, 1600)
        self.assertEqual(path.read_bytes(), original)

    def test_profile_rejects_foreign_engine(self):
        torch.save({"format_version": 2, "engine": "other"}, self.root / "foreign.pt")
        with self.assertRaisesRegex(ValueError, "unsupported voice engine"):
            profile_manager.load_profile("foreign")

    def test_omnivoice_loader_uses_mps_fp16_without_extra_asr(self):
        model = SimpleNamespace(audio_tokenizer=SimpleNamespace(device="cpu"))
        with (
            patch.dict(os.environ),
            patch.object(voice_cloner, "_tts_model", None),
            patch.object(voice_cloner, "DEVICE", "mps"),
            patch.object(voice_cloner, "TORCH_DTYPE_STR", "float16"),
            patch("omnivoice.OmniVoice.from_pretrained", return_value=model) as load,
        ):
            self.assertIs(voice_cloner._get_tts_model(), model)
            self.assertIs(voice_cloner._get_tts_model(), model)
            self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "1")
        load.assert_called_once_with(
            "k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16,
            attn_implementation="sdpa", load_asr=False,
        )
        self.assertEqual(model.audio_tokenizer.device, "cpu")

    def test_omnivoice_prompt_receives_supplied_transcript_and_mono_tensor(self):
        from unittest.mock import Mock
        model = Mock()
        model.create_voice_clone_prompt.return_value = self.prompt()
        with patch.object(voice_cloner, "_get_tts_model", return_value=model):
            prompt = voice_cloner.create_clone_prompt_from_numpy(
                np.full((1600, 2), 327, dtype=np.int16), 16000, " words ",
            )
        self.assertIsInstance(prompt, VoiceClonePrompt)
        args = model.create_voice_clone_prompt.call_args.kwargs
        self.assertEqual(args["ref_text"], "words")
        self.assertEqual(set(args), {"ref_audio", "ref_text"})
        wav, sr = args["ref_audio"]
        self.assertIsInstance(wav, torch.Tensor)
        self.assertEqual(wav.shape, (1600,))
        self.assertEqual(sr, 16000)

    def test_omnivoice_generation_uses_native_api_and_model_sample_rate(self):
        from unittest.mock import Mock
        model = Mock(sampling_rate=24000)
        model.generate.return_value = [np.full(240, 0.1)]
        prompt = self.prompt()
        with (
            patch.object(voice_cloner, "_get_tts_model", return_value=model),
            patch.object(voice_cloner, "DEVICE", "cpu"),
        ):
            wav, sr = voice_cloner.generate_speech("Hello", "Auto", prompt)
            direct_wav, direct_sr = voice_cloner.generate_speech_direct(
                "こんにちは", "Japanese", "reference.wav", " words ",
            )
        self.assertEqual(sr, 24000)
        self.assertEqual(direct_sr, 24000)
        self.assertEqual(wav.dtype, np.float32)
        self.assertTrue(np.isfinite(direct_wav).all())
        self.assertEqual(model.generate.call_args_list[0].kwargs, {
            "text": "Hello", "language": None, "voice_clone_prompt": prompt,
            "num_step": 32, "speed": 1.0,
        })
        self.assertEqual(model.generate.call_args_list[1].kwargs, {
            "text": "こんにちは", "language": "Japanese", "ref_audio": "reference.wav",
            "ref_text": "words", "num_step": 32, "speed": 1.0,
        })

    def test_empty_transcript_never_triggers_omnivoice_auto_asr(self):
        with patch.object(voice_cloner, "_get_tts_model") as load:
            with self.assertRaisesRegex(ValueError, "transcript"):
                voice_cloner.create_clone_prompt("reference.wav", " ")
            with self.assertRaisesRegex(ValueError, "transcript"):
                voice_cloner.generate_speech_direct("Hello", "English", "reference.wav", "")
        load.assert_not_called()

    def test_omnivoice_language_choices_are_not_limited_to_previous_engine(self):
        languages = voice_cloner.get_supported_languages()
        self.assertGreater(len(languages), 600)
        self.assertEqual(languages[0], "Auto")
        self.assertIn("Japanese", languages)
        self.assertIn("English", languages)

    def test_profile_load_and_delete_reject_path_traversal(self):
        for name in ("../outside", "/tmp/outside", "..\\outside", "", ".."):
            for operation in (profile_manager.load_profile, profile_manager.delete_profile):
                with self.subTest(name=name, operation=operation.__name__):
                    with self.assertRaises(ValueError):
                        operation(name)

    def test_profile_symlinks_are_not_loaded_listed_or_deleted(self):
        target = self.root / "target.dat"
        target.write_text("unchanged")
        (self.root / "linked.pt").symlink_to(target)
        self.assertNotIn("linked", profile_manager.list_profiles())
        for operation in (profile_manager.load_profile, profile_manager.delete_profile):
            with self.assertRaises(ValueError):
                operation("linked")
        self.assertEqual(target.read_text(), "unchanged")

    def test_failed_profile_write_keeps_existing_profile(self):
        saved = profile_manager.save_profile("speaker", self.prompt(), "words")
        previous = saved.read_bytes()
        with patch.object(profile_manager.torch, "save", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                profile_manager.save_profile("speaker", self.prompt(), "new words")
        self.assertEqual(saved.read_bytes(), previous)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_saved_dropdown_uses_sanitized_filename(self):
        with patch.object(app, "create_clone_prompt", return_value=self.prompt()):
            status, dropdown, prompt = app.on_save_profile("Voice / A", "words", str(self.audio()))
        self.assertIn("saved", status)
        self.assertEqual(dropdown["value"], "Voice _ A")
        self.assertIn(dropdown["value"], profile_manager.list_profiles())
        self.assertIsNotNone(prompt)

    def test_uploads_use_independent_files_and_invalidate_cached_prompt(self):
        source = self.audio(rate=44100, channels=2)
        with patch.object(app, "transcribe_audio", return_value="words"):
            first = app.on_audio_file_upload(str(source), "Auto")
            second = app.on_audio_file_upload(str(source), "Auto")
        self.assertNotEqual(first[0], second[0])
        self.assertTrue(Path(first[0]).exists())
        info = sf.info(first[0])
        self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
        self.assertEqual(len(first), 5)
        self.assertIsNone(first[-1])

    def test_mic_keeps_quiet_integer_audio_at_its_original_level(self):
        samples = np.full((1600, 2), 327, dtype=np.int16)
        with patch.object(app, "transcribe_audio", return_value="words"):
            result = app.on_mic_recording_stop((16000, samples), "Auto")
        self.assertIsNone(result[-1])
        wav, sr = sf.read(result[0])
        self.assertEqual(sr, 16000)
        self.assertAlmostEqual(float(wav.max()), 327 / 32768, places=4)

    def test_empty_reference_inputs_also_clear_cached_prompt(self):
        for result in (
            app.on_fetch_youtube("", "Auto"),
            app.on_mic_recording_stop(None, "Auto"),
            app.on_audio_file_upload(None, "Auto"),
        ):
            self.assertEqual(len(result), 5)
            self.assertIsNone(result[-1])

    def test_waveform_edit_retranscribes_and_invalidates_old_reference(self):
        source = self.root / "full.wav"
        signal = np.linspace(-0.5, 0.5, 16 * 16000)
        sf.write(source, signal, 16000, subtype="FLOAT")
        edited = self.root / "edited.wav"
        selected = signal[6 * 16000:9 * 16000]
        sf.write(edited, selected, 16000, subtype="FLOAT")
        session_id = "c" * 32
        with patch.object(app, "transcribe_audio", return_value="selected words") as transcribe:
            steps = list(app.on_reference_audio_change(str(edited), str(source), "English", session_id))
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][0], "")
        self.assertIsNone(steps[0][2])
        self.assertIsNone(steps[0][3])
        text, _, path, prompt = steps[-1]
        self.assertEqual(text, "selected words")
        self.assertIsNone(prompt)
        self.assertEqual(Path(path).parent, app._reference_directory(session_id))
        audio, sr = sf.read(path)
        self.assertEqual(sr, 16000)
        np.testing.assert_allclose(audio, selected, atol=1 / 32768)
        self.assertAlmostEqual(sf.info(source).duration, 16.0)
        transcribe.assert_called_once_with(Path(path), language="English")

    def test_backend_audio_cache_copy_does_not_transcribe_again(self):
        source = self.audio()
        cached = self.root / "gradio-cache.wav"
        shutil.copyfile(source, cached)
        with patch.object(app, "on_audio_file_upload") as process:
            steps = list(app.on_reference_audio_change(str(cached), str(source), "Auto"))
        process.assert_not_called()
        self.assertEqual(steps, [tuple({"__type__": "update"} for _ in range(4))])

    def test_clearing_waveform_removes_reference_and_prompt(self):
        steps = list(app.on_reference_audio_change(None, str(self.audio()), "Auto"))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0][0], "")
        self.assertIsNone(steps[0][2])
        self.assertIsNone(steps[0][3])

    def test_backend_clear_keeps_the_source_error_message(self):
        steps = list(app.on_reference_audio_change(None, None, "Auto"))
        self.assertEqual(steps, [tuple({"__type__": "update"} for _ in range(4))])

    def test_waveform_reset_can_restore_original_after_trim(self):
        source = self.audio(rate=16000)
        edited = self.root / "short.wav"
        sf.write(edited, np.zeros(500), 16000)
        with patch.object(app, "transcribe_audio", return_value="restored words"):
            trimmed = list(app.on_reference_audio_change(str(edited), str(source), "Auto"))[-1]
            restored = list(app.on_reference_audio_change(str(source), trimmed[2], "Auto"))[-1]
        self.assertEqual(restored[0], "restored words")
        self.assertAlmostEqual(sf.info(restored[2]).duration, sf.info(source).duration)

    def test_stale_prompt_from_overlapping_save_is_not_used(self):
        stale = app._cache_prompt(self.prompt(), "reference_a.wav", "old words")
        with (
            patch.object(app, "generate_speech") as cached,
            patch("voice_clone_webui.voice_cloner.generate_speech_direct", return_value=(np.zeros(10), 16000)) as direct,
        ):
            result = app.on_generate_speech("hello", "English", stale, "reference_b.wav", "new words")
        cached.assert_not_called()
        self.assertEqual(direct.call_args.kwargs["ref_audio_path"], "reference_b.wav")
        self.assertIn("complete", result[1])

    def test_transcript_edit_prevents_cached_prompt_reuse(self):
        stale = app._cache_prompt(self.prompt(), "reference.wav", "old words")
        with (
            patch.object(app, "generate_speech") as cached,
            patch("voice_clone_webui.voice_cloner.generate_speech_direct", return_value=(np.zeros(10), 16000)) as direct,
        ):
            app.on_generate_speech("hello", "English", stale, "reference.wav", "edited words")
        cached.assert_not_called()
        self.assertEqual(direct.call_args.kwargs["ref_text"], "edited words")

    def test_matching_reference_reuses_cached_prompt(self):
        prompt = self.prompt()
        state = app._cache_prompt(prompt, "reference.wav", "words")
        with patch.object(app, "generate_speech", return_value=(np.zeros(10), 16000)) as cached:
            app.on_generate_speech("hello", "English", state, "reference.wav", "words")
        self.assertIs(cached.call_args.kwargs["voice_clone_prompt"], prompt)

    def test_session_cleanup_removes_only_its_own_audio(self):
        first = "a" * 32
        second = "b" * 32
        first_dir = app._reference_directory(first)
        second_dir = app._reference_directory(second)
        (first_dir / "voice.wav").write_bytes(b"first")
        other = second_dir / "voice.wav"
        other.write_bytes(b"second")
        app._cleanup_session("../outside")
        self.assertTrue(other.exists())
        app._cleanup_session(first)
        self.assertFalse(first_dir.exists())
        self.assertEqual(other.read_bytes(), b"second")

    def test_whisper_accepts_pcm16_mono_without_conversion(self):
        source = self.audio()
        self.assertEqual(transcriber._ensure_16khz(source), source)

    def test_whisper_converts_float_stereo_even_at_16khz(self):
        source = self.audio(channels=2, subtype="FLOAT")
        result = transcriber._ensure_16khz(source)
        info = sf.info(result)
        self.assertNotEqual(result, source)
        self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))

    def test_whisper_cleans_converted_audio_on_success_and_failure(self):
        source = self.audio(rate=48000)
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: [SimpleNamespace(text="words")])
        with patch.object(transcriber, "_get_whisper_model", return_value=model):
            self.assertEqual(transcriber.transcribe_audio(source), "words")
        self.assertEqual(list(self.root.glob("whisper_*.wav")), [])
        with patch.object(transcriber, "_get_whisper_model", side_effect=RuntimeError("model unavailable")):
            with self.assertRaises(RuntimeError):
                transcriber.transcribe_audio(source)
        self.assertEqual(list(self.root.glob("whisper_*.wav")), [])
        self.assertTrue(source.exists())

    def test_youtube_timestamp_and_query_encoding(self):
        url = "https://youtu.be/example?v=abc&list=a%26b&t=1h2m3s#t=9"
        self.assertEqual(audio_downloader._parse_start_seconds(url), 3723)
        self.assertEqual(audio_downloader._strip_time_param(url), "https://youtu.be/example?v=abc&list=a%26b")
        self.assertEqual(audio_downloader._parse_start_seconds("https://youtu.be/a?t=12mgarbage"), 0)

    def test_youtube_rejects_non_youtube_urls_before_running_a_process(self):
        with patch.object(audio_downloader.subprocess, "run") as run:
            for url in ("--exec=touch", "file:///tmp/a", "https://youtube.com.evil.test/a", "http://localhost/a"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    audio_downloader.download_youtube_audio(url)
            run.assert_not_called()

    def test_youtube_downloads_use_separate_request_directories(self):
        def run(command, **kwargs):
            if "yt_dlp" in command:
                template = command[command.index("-o") + 1]
                sf.write(template.replace("%(ext)s", "wav"), np.zeros(48000), 16000)
                self.assertEqual(command[-2], "--")
            else:
                sf.write(command[-1], np.zeros(48000), 16000)
            return SimpleNamespace(returncode=0, stderr="")

        with patch.object(audio_downloader.subprocess, "run", side_effect=run):
            first, _ = audio_downloader.download_youtube_audio("https://youtu.be/example?t=2")
            second, _ = audio_downloader.download_youtube_audio("https://youtu.be/example?t=5")
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_youtube_windows_are_centered_and_clamped_at_video_start(self):
        real_run = subprocess.run
        commands = []

        def run(command, **kwargs):
            if "yt_dlp" not in command:
                return real_run(command, **kwargs)
            commands.append(command)
            template = command[command.index("-o") + 1]
            sf.write(template.replace("%(ext)s", "wav"), np.zeros(20 * 16000), 16000)
            return SimpleNamespace(returncode=0, stderr="")

        cases = (
            ("?t=1018", "*16:50-17:06", 16),
            ("?t=3", "*0:00-0:11", 11),
            ("?t=0", "*0:00-0:08", 8),
            ("", "*0:00-0:08", 8),
            ("#t=1h2m3s", "*61:55-62:11", 16),
        )
        with patch.object(audio_downloader.subprocess, "run", side_effect=run):
            for suffix, section, expected_duration in cases:
                with self.subTest(url=suffix):
                    output, _ = audio_downloader.download_youtube_audio("https://youtu.be/example" + suffix)
                    command = commands[-1]
                    self.assertEqual(command[command.index("--download-sections") + 1], section)
                    self.assertEqual(command[command.index("-f") + 1], "bestaudio")
                    self.assertIn("--force-keyframes-at-cuts", command)
                    self.assertAlmostEqual(sf.info(output).duration, expected_duration, places=3)

    def test_youtube_video_end_uses_actual_available_duration(self):
        real_run = subprocess.run

        def run(command, **kwargs):
            if "yt_dlp" not in command:
                return real_run(command, **kwargs)
            template = command[command.index("-o") + 1]
            sf.write(template.replace("%(ext)s", "wav"), np.zeros(20000), 16000)
            return SimpleNamespace(returncode=0, stderr="")

        with patch.object(audio_downloader.subprocess, "run", side_effect=run):
            output, status = audio_downloader.download_youtube_audio("https://youtu.be/example?t=100")
        self.assertAlmostEqual(sf.info(output).duration, 1.25)
        self.assertIn("1.25 seconds available", status)

    def test_ui_source_events_reset_prompt_and_mic_runs_only_once(self):
        demo = app.build_ui()
        self.addCleanup(demo.close)
        callbacks = list(demo.fns.values())
        sources = [fn for fn in callbacks if fn.fn in {
            app.on_fetch_youtube, app.on_mic_recording_stop, app.on_audio_file_upload,
        }]
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(len(fn.outputs) == 5 for fn in sources))
        self.assertTrue(all(fn.outputs[-1] is sources[0].outputs[-1] for fn in sources))
        self.assertTrue(all(fn.inputs[-1].delete_callback is app._cleanup_session for fn in sources))
        edits = [fn for fn in callbacks if any(event == "input" for _, event in fn.targets)]
        self.assertEqual(len(edits), 1)
        self.assertIs(edits[0].outputs[0], sources[0].outputs[-1])
        self.assertIsNone(edits[0].fn())
        waveform_edits = [fn for fn in callbacks if fn.fn is app.on_reference_audio_change]
        self.assertEqual(len(waveform_edits), 1)
        waveform = waveform_edits[0].inputs[0]
        self.assertTrue(waveform.interactive)
        self.assertTrue(waveform.editable)
        self.assertEqual(waveform.waveform_options.sample_rate, 16000)
        self.assertNotIn(waveform, waveform_edits[0].outputs)
        self.assertEqual(waveform_edits[0].concurrency_id, sources[0].concurrency_id)


if __name__ == "__main__":
    unittest.main()
