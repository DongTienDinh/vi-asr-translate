"""Gradio Web Demonstration Interface for Vietnamese ASR -> English Translation (Section 26)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def format_seconds(seconds: float) -> str:
    """Format seconds into MM:SS timestamp."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def build_demo(pipeline_fn: Optional[Any] = None):
    """Construct Gradio UI for speech recognition and translation."""
    import gradio as gr

    def process_audio(
        audio_file: Optional[str],
        audio_mic: Optional[str],
        enable_translation: bool,
    ) -> Tuple[str, str, List[List[Any]]]:
        audio_input = audio_file or audio_mic
        if not audio_input:
            return "Vui lòng chọn hoặc ghi âm file âm thanh.", "", []

        if pipeline_fn is not None:
            segments = pipeline_fn(audio_input, translate=enable_translation)
        else:
            # Fallback mock for UI demonstration when weights are not yet loaded
            segments = [
                {
                    "segment_id": 1,
                    "start_sec": 0.0,
                    "end_sec": 3.2,
                    "vietnamese": "xin chào việt nam",
                    "english": "Hello Vietnam",
                },
                {
                    "segment_id": 2,
                    "start_sec": 3.5,
                    "end_sec": 7.8,
                    "vietnamese": "chào mừng bạn đến với hệ thống nhận dạng tiếng việt",
                    "english": "Welcome to the Vietnamese speech recognition system",
                },
            ]

        full_vi = " ".join([s.get("vietnamese", "") for s in segments]).strip()
        full_en = " ".join([s.get("english", "") for s in segments]).strip()

        table = []
        for s in segments:
            table.append([
                s.get("segment_id", 0),
                format_seconds(s.get("start_sec", 0.0)),
                format_seconds(s.get("end_sec", 0.0)),
                s.get("vietnamese", ""),
                s.get("english", "") if enable_translation else "-",
            ])

        return full_vi, full_en, table

    with gr.Blocks(title="Vietnamese ASR → English Translation (ViP-VL + VinAI)") as demo:
        gr.Markdown(
            """
            # 🎙️ Vietnamese ASR → English Translation
            ### Kiến trúc ViP-VL ChunkFormer + Token–Frame Attention Pooling & VinAI Translate
            *Hỗ trợ xử lý tệp âm thanh dài và micro trực tiếp với VAD phân đoạn tự động.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("📁 Tải tệp âm thanh (Upload File)"):
                    file_input = gr.Audio(sources=["upload"], type="filepath", label="Chọn tệp âm thanh")
                with gr.Tab("🎤 Ghi âm (Microphone)"):
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Ghi âm trực tiếp")

                translate_checkbox = gr.Checkbox(
                    value=True,
                    label="Bật dịch tự động sang tiếng Anh (VinAI Translate vi2en)",
                )

                submit_btn = gr.Button("🚀 Nhận dạng & Dịch thuật", variant="primary", size="lg")

            with gr.Column(scale=1):
                out_vi = gr.Textbox(
                    label="🇻🇳 Văn bản tiếng Việt (Vietnamese Transcript)",
                    lines=3,
                )
                out_en = gr.Textbox(
                    label="🇬🇧 Bản dịch tiếng Anh (English Translation)",
                    lines=3,
                )

        gr.Markdown("### ⏱️ Chi tiết từng phân đoạn câu (Timed Segments)")
        segments_table = gr.Dataframe(
            headers=["Phân đoạn #", "Bắt đầu", "Kết thúc", "Tiếng Việt", "Tiếng Anh"],
            datatype=["number", "str", "str", "str", "str"],
            row_count=3,
            column_count=(5, "fixed"),
        )

        submit_btn.click(
            fn=process_audio,
            inputs=[file_input, mic_input, translate_checkbox],
            outputs=[out_vi, out_en, segments_table],
        )

    return demo


def launch_demo(
    pipeline: Optional[Any] = None,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio demo web application."""
    demo = build_demo(pipeline_fn=pipeline.process_file if pipeline else None)
    demo.launch(server_port=port, share=share)
