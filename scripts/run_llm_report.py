from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT_FOR_IMPORT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.llm.rag_reporter import TunnelRAGReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="result.xlsx + 터널 시방서 TXT(RAG) 기반 최종 보고서 생성"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="프로젝트 루트 경로 (기본값: 현재 폴더)",
    )
    parser.add_argument(
        "--result-excel",
        default="data/outputs/result.xlsx",
        help="result.xlsx 경로",
    )
    parser.add_argument(
        "--specs-dir",
        default="data/knowledge/specs_selected_txt",
        help="선별된 시방서 TXT 폴더 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/llm_report",
        help="최종 보고서 저장 폴더",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Responses API에 사용할 모델명",
    )
    parser.add_argument(
        "--vector-store-name",
        default="tunnel-method-specs",
        help="재사용할 벡터스토어 이름",
    )
    parser.add_argument(
        "--top-k-methods",
        type=int,
        default=3,
        help="보고서에서 중점적으로 비교할 상위 공법 개수",
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=6,
        help="file_search 최대 검색 결과 수",
    )
    parser.add_argument(
        "--no-reuse-vector-store",
        action="store_true",
        help="기존 동일 이름의 벡터스토어를 재사용하지 않고 새로 생성",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    reporter = TunnelRAGReporter(
        model=args.model,
        result_excel_path=project_root / args.result_excel,
        specs_dir=project_root / args.specs_dir,
        output_dir=project_root / args.output_dir,
        project_root=project_root,
        vector_store_name=args.vector_store_name,
        top_k_methods=args.top_k_methods,
        max_search_results=args.max_search_results,
        reuse_existing_vector_store=not args.no_reuse_vector_store,
    )

    artifacts = reporter.run()

    print("최종 보고서 생성 완료")
    print(f"- vector_store_id: {artifacts.vector_store_id}")
    print(f"- markdown: {artifacts.markdown_path}")
    print(f"- json: {artifacts.json_path}")


if __name__ == "__main__":
    main()
