"""Main entry point for CV-to-PowerPoint automation system."""
import argparse
import sys
from pathlib import Path

from pipeline.orchestrator import CVAutomationPipeline
from utils.logger import logger


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='CV-to-PowerPoint Automation System (DETERMINISTIC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m src.main --cv resume.pdf --template template.pptx
  python -m src.main --cv resume.docx --template template.pptx --output result.pptx
  python -m src.main --cv resume.txt --template template.pptx --trace

Requirements:
  - GEMINI_API_KEY must be set in .env file
  - Temperature is hardcoded to 0.0 for deterministic output
        '''
    )

    parser.add_argument(
        '--cv',
        required=True,
        help='Path to CV file (.pdf, .docx, or .txt)'
    )

    parser.add_argument(
        '--template',
        required=True,
        help='Path to PowerPoint template (.pptx)'
    )

    parser.add_argument(
        '--output',
        help='Path for output presentation (default: auto-generated in outputs/)'
    )

    parser.add_argument(
        '--trace',
        action='store_true',
        help='Display traceability information'
    )

    args = parser.parse_args()

    # Verify files exist
    cv_path = Path(args.cv)
    template_path = Path(args.template)

    if not cv_path.exists():
        logger.error(f"CV file not found: {cv_path}")
        sys.exit(1)

    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        sys.exit(1)

    # Execute pipeline
    logger.info("=" * 80)
    logger.info("CV-to-PowerPoint Automation System - DETERMINISTIC MODE")
    logger.info("=" * 80)
    logger.info(f"CV File: {cv_path}")
    logger.info(f"Template: {template_path}")
    logger.info(f"Deterministic: temperature=0.0, top_k=1, top_p=0.1")
    logger.info("=" * 80)

    try:
        pipeline = CVAutomationPipeline()

        result = pipeline.execute(
            cv_file_path=str(cv_path),
            template_path=str(template_path),
            output_path=args.output
        )

        if result["status"] == "success":
            logger.info("=" * 80)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Output File: {result['output_file']}")
            logger.info(f"Traceability JSON: {result['traceability_json']}")
            logger.info(f"Traceability Excel: {result['traceability_excel']}")
            logger.info("")
            logger.info("Summary:")
            logger.info(f"  - Total Placeholders: {result['summary']['total_placeholders']}")
            logger.info(f"  - Mapped Placeholders: {result['summary']['mapped_placeholders']}")
            logger.info(f"  - Transformations: {result['summary']['transformations']}")
            logger.info(f"  - Full Compliance: {result['summary']['full_compliance']}")
            logger.info(f"  - Partial Compliance: {result['summary']['partial_compliance']}")
            logger.info("=" * 80)

            if args.trace:
                logger.info("\nTraceability Matrix:")
                for stage in result["stages"]:
                    logger.info(f"  [{stage['status']}] {stage['stage_name']}")

            sys.exit(0)
        else:
            logger.error("=" * 80)
            logger.error("✗ PIPELINE FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            logger.error(f"Error Type: {result.get('error_type', 'Unknown')}")

            if args.trace and 'stages' in result:
                logger.error("\nStage Status:")
                for stage in result["stages"]:
                    logger.error(f"  [{stage['status']}] {stage['stage_name']}")

            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
