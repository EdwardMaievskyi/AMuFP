from src.process_and_extract import (get_file_type, extract_doc_content,
                                     extract_docx_content,
                                     extract_pptx_content,
                                     extract_pdf_content,
                                     extract_image_content,
                                     extract_text_content,
                                     extract_csv_tsv_content,
                                     extract_excel_content,
                                     extract_odt_content,
                                     extract_ods_content, extract_odp_content)
from langgraph.graph import StateGraph, END
from src.state_models import FileProcessingState
from src.utils.logger import get_file_processing_logger

# Initialize logger at module level
logger = get_file_processing_logger()


def decide_extraction_path(state: FileProcessingState):
    """Router function to direct the workflow based on file type."""
    mime_type = state.file_type or ''
    logger.info("Deciding extraction path for mime type: %s", mime_type)

    if 'image' in mime_type:
        return "extract_image"
    if 'pdf' in mime_type:
        return "extract_pdf"
    if 'plain' in mime_type or 'markdown' in mime_type:
        return "extract_text"
    if 'csv' in mime_type or 'tab-separated' in mime_type:
        return "extract_csv_tsv"
    if 'ms-excel' in mime_type or 'spreadsheetml' in mime_type:
        return "extract_excel"
    if 'msword' in mime_type:
        return "extract_doc"
    if 'wordprocessingml' in mime_type:
        return "extract_docx"
    if 'presentationml' in mime_type:
        return "extract_pptx"
    if 'opendocument.text' in mime_type:
        return "extract_odt"
    if 'opendocument.spreadsheet' in mime_type:
        return "extract_ods"
    if 'opendocument.presentation' in mime_type:
        return "extract_odp"
    logger.warning("No specific extraction path for %s, ending subgraph.",
                   mime_type)
    return END


sub_workflow = StateGraph(FileProcessingState)
sub_workflow.add_node("get_file_type", get_file_type)
sub_workflow.add_node("extract_doc", extract_doc_content)
sub_workflow.add_node("extract_docx", extract_docx_content)
sub_workflow.add_node("extract_pptx", extract_pptx_content)
sub_workflow.add_node("extract_pdf", extract_pdf_content)
sub_workflow.add_node("extract_image", extract_image_content)
sub_workflow.add_node("extract_text",
                      extract_text_content)
sub_workflow.add_node("extract_csv_tsv", extract_csv_tsv_content)
sub_workflow.add_node("extract_excel", extract_excel_content)
sub_workflow.add_node("extract_odt", extract_odt_content)
sub_workflow.add_node("extract_ods", extract_ods_content)
sub_workflow.add_node("extract_odp", extract_odp_content)

sub_workflow.set_entry_point("get_file_type")
sub_workflow.add_conditional_edges("get_file_type",
                                   decide_extraction_path)

for node_name in sub_workflow.nodes:
    if node_name != "get_file_type":
        sub_workflow.add_edge(node_name, END)

file_processing_graph = sub_workflow.compile()
