from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


@dataclass
class ReportArtifacts:
    markdown_path: Path
    json_path: Path
    response_text: str
    vector_store_id: str


class TunnelRAGReporter:
    """Read result.xlsx, search spec text files, and generate a final report with the Responses API."""

    def __init__(
        self,
        *,
        model: str,
        result_excel_path: str | Path,
        specs_dir: str | Path,
        output_dir: str | Path,
        project_root: str | Path | None = None,
        vector_store_name: str = "tunnel-method-specs",
        top_k_methods: int = 3,
        max_search_results: int = 12,
        reuse_existing_vector_store: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.result_excel_path = Path(result_excel_path)
        self.specs_dir = Path(specs_dir)
        self.output_dir = Path(output_dir)
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.vector_store_name = vector_store_name
        self.top_k_methods = top_k_methods
        self.max_search_results = max_search_results
        self.reuse_existing_vector_store = reuse_existing_vector_store

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> ReportArtifacts:
        input_df, result_df = self._load_excel()
        vector_store_id = self._prepare_vector_store()
        prompt = self._build_prompt(input_df=input_df, result_df=result_df)
        response = self._ask_model(prompt=prompt, vector_store_id=vector_store_id)
        markdown_path, json_path = self._save_outputs(
            input_df=input_df,
            result_df=result_df,
            response=response,
            vector_store_id=vector_store_id,
        )

        return ReportArtifacts(
            markdown_path=markdown_path,
            json_path=json_path,
            response_text=response.output_text,
            vector_store_id=vector_store_id,
        )

    def _load_excel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.result_excel_path.exists():
            raise FileNotFoundError(f"result.xlsx 파일을 찾을 수 없습니다: {self.result_excel_path}")

        xls = pd.ExcelFile(self.result_excel_path)

        required_sheets = {"input_data", "result"}
        missing_sheets = required_sheets - set(xls.sheet_names)
        if missing_sheets:
            raise ValueError(f"필수 시트가 없습니다: {sorted(missing_sheets)}")

        input_df = pd.read_excel(self.result_excel_path, sheet_name="input_data")
        result_df = pd.read_excel(self.result_excel_path, sheet_name="result")

        expected_cols = {"공법", "raw_score", "점수(100점 만점)", "판정"}
        missing_cols = expected_cols - set(result_df.columns)
        if missing_cols:
            raise ValueError(f"result 시트 필수 컬럼이 없습니다: {sorted(missing_cols)}")

        result_df = result_df.sort_values(by="점수(100점 만점)", ascending=False).reset_index(drop=True)
        return input_df, result_df

    def _prepare_vector_store(self) -> str:
        if not self.specs_dir.exists():
            raise FileNotFoundError(f"시방서 텍스트 폴더를 찾을 수 없습니다: {self.specs_dir}")

        txt_files = sorted(self.specs_dir.glob("*.txt"))
        txt_files = [p for p in txt_files if p.name.lower() != "readme.txt"]

        if not txt_files:
            raise FileNotFoundError(f"업로드할 TXT 파일이 없습니다: {self.specs_dir}")

        if self.reuse_existing_vector_store:
            existing_id = self._find_vector_store_by_name(self.vector_store_name)
            if existing_id:
                return existing_id

        vector_store = self.client.vector_stores.create(name=self.vector_store_name)

        file_streams = [open(path, "rb") for path in txt_files]
        try:
            batch = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=file_streams,
            )
        finally:
            for stream in file_streams:
                stream.close()

        status = getattr(batch, "status", None)
        if status != "completed":
            raise RuntimeError(f"벡터스토어 파일 처리 실패 또는 미완료 상태입니다. status={status}")

        return vector_store.id

    def _find_vector_store_by_name(self, name: str) -> str | None:
        page = self.client.vector_stores.list(limit=100)
        stores = list(getattr(page, "data", []))

        while True:
            for store in stores:
                if getattr(store, "name", None) == name:
                    return store.id

            if not getattr(page, "has_next_page", lambda: False)():
                break

            page = page.get_next_page()
            stores = list(getattr(page, "data", []))

        return None

    def _forced_method_context(self, result_df: pd.DataFrame) -> str:
        """
        침매식이 정량 결과 기준 1위 또는 2위일 때만
        침매공법 관련 txt를 직접 읽어서 prompt에 넣는다.
        file_search가 침매 문서를 못 잡는 문제를 보완하기 위한 강제 참조 로직.
        """
        top2_methods = result_df.head(2)["공법"].astype(str).tolist()
        should_read_immersed = any("침매" in method for method in top2_methods)

        if not should_read_immersed:
            return ""

        keywords = ["침매", "KOSHA_C_89", "immersed", "immersion"]

        contexts: list[str] = []

        for path in sorted(self.specs_dir.glob("*.txt")):
            filename_lower = path.name.lower()

            if not any(keyword.lower() in filename_lower for keyword in keywords):
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")

            contexts.append(
                f"[강제 참조 문서: {path.name}]\n"
                f"{text[:8000]}"
            )

        return "\n\n".join(contexts)

    def _build_prompt(self, *, input_df: pd.DataFrame, result_df: pd.DataFrame) -> str:
        project_conditions = self._project_conditions_text(input_df)
        result_summary = self._result_summary_text(result_df)
        top_methods = result_df.head(self.top_k_methods)["공법"].tolist()
        forced_context = self._forced_method_context(result_df)

        return f"""
너는 '터널 공법 추천 결과를 해석하는 기술검토 전문가'다.

중요 역할:
- 정량 점수(result.xlsx)는 이미 계산 완료된 결과이므로, 너는 점수를 다시 계산하지 않는다.
- 너의 역할은 result.xlsx 결과를 해석하고, 첨부된 터널 기준/시방서 텍스트를 검색하여 근거를 붙이는 것이다.
- 단, 아래 '강제 참조 문서'가 제공된 경우에는 file_search 검색 결과보다 우선적으로 해당 문서를 검토해야 한다.
- 문서에 없는 내용은 추정이라고 분명히 밝혀라.

프로젝트 입력조건:
{project_conditions}

정량 결과 요약:
{result_summary}

침매식 강제 참조 문서:
{forced_context if forced_context else "해당 없음"}

상위 후보 공법:
{json.dumps(top_methods, ensure_ascii=False)}

아래 형식으로 한국어 보고서를 작성해라.

# 터널 공법 최종 검토 보고서

## 1. 프로젝트 조건 요약
- 입력 조건을 간단히 정리

## 2. 정량 평가 결과 해석
- 상위 공법 {self.top_k_methods}개를 비교
- 점수 차이와 판정을 해석
- 침매식이 1위 또는 2위이고 강제 참조 문서가 제공된 경우, 침매공법 근거를 반드시 반영

## 3. 시방서/기준 기반 검토사항
- 업로드된 문서에서 실제로 검색한 근거 중심으로 작성
- 강제 참조 문서가 제공된 경우, 해당 문서 내용을 우선 반영
- 공법별 설계/시공상 핵심 검토사항 정리
- 관련 문서명도 함께 언급

## 4. 최종 추천 공법
- 추천 공법 1개 제시
- 추천 사유를 정량 결과 + 문서 근거 관점에서 설명

## 5. 추가 확인 필요사항
- 현장 조사, 지반조사, 수압, 민감 구조물, 시공성, 장비 조달 등 후속 확인사항

## 6. 한계
- 이번 보고서의 한계를 솔직히 작성
- 강제 참조 문서가 없거나 검색되지 않은 공법은 문서 근거가 제한적이라고 명시

추가 지침:
- 표준시방서/설계기준 문구를 길게 복붙하지 말고 요약해서 설명
- 근거가 있는 경우 파일명 수준으로 출처를 본문에 괄호로 표시
- 문서 검색 결과와 강제 참조 문서를 우선하고, 검색되지 않은 내용은 단정하지 말 것
""".strip()

    def _project_conditions_text(self, input_df: pd.DataFrame) -> str:
        row = input_df.iloc[0].fillna("")
        lines = []
        for col in input_df.columns:
            lines.append(f"- {col}: {row[col]}")
        return "\n".join(lines)

    def _result_summary_text(self, result_df: pd.DataFrame) -> str:
        lines = []
        for idx, row in result_df.iterrows():
            remark = "" if pd.isna(row.get("비고")) else f", 비고: {row.get('비고')}"
            lines.append(
                f"{idx + 1}. 공법: {row['공법']}, "
                f"raw_score: {row['raw_score']:.4f}, "
                f"점수: {row['점수(100점 만점)']}, "
                f"판정: {row['판정']}{remark}"
            )
        return "\n".join(lines)

    def _ask_model(self, *, prompt: str, vector_store_id: str) -> Any:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": self.max_search_results,
                }
            ],
            include=["file_search_call.results"],
        )
        return response

    def _save_outputs(
        self,
        *,
        input_df: pd.DataFrame,
        result_df: pd.DataFrame,
        response: Any,
        vector_store_id: str,
    ) -> tuple[Path, Path]:
        markdown_path = self.output_dir / "final_report.md"
        json_path = self.output_dir / "final_report.json"

        search_results = self._extract_search_results(response)
        payload = {
            "vector_store_id": vector_store_id,
            "model": self.model,
            "project_input": input_df.to_dict(orient="records"),
            "result_table": result_df.to_dict(orient="records"),
            "search_results": search_results,
            "report_text": response.output_text,
        }

        markdown_path.write_text(response.output_text, encoding="utf-8")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return markdown_path, json_path

    def _extract_search_results(self, response: Any) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        output_items = getattr(response, "output", []) or []

        for item in output_items:
            if getattr(item, "type", None) != "file_search_call":
                continue

            results = getattr(item, "results", []) or []
            for result in results:
                text = getattr(result, "text", None)
                filename = getattr(result, "filename", None)
                score = getattr(result, "score", None)

                extracted.append(
                    {
                        "filename": filename,
                        "score": score,
                        "text": text,
                    }
                )

        return extracted