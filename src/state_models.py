from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class FileSource(str, Enum):
    """Enumeration for file source types."""
    LOCAL_PATH = "local_path"
    URL = "url"


class FileMetaData(BaseModel):
    """Metadata for file inputs including path/URL and source type."""
    file_path_or_url: Optional[str] = Field(
        default=None,
        description="File path or URL to the image or document"
    )
    file_source: Optional[FileSource] = Field(
        default=None,
        description=("Source type of the file (local_path or url). " +
                     "Auto-inferred from file_path_or_url if not provided.")
    )

    @field_validator('file_path_or_url')
    @classmethod
    def validate_file_path_or_url(cls, v):
        """Validate that the file path or URL is not empty and has proper
        format.
        """
        if v is None:
            return v

        if not v or not v.strip():
            raise ValueError("File path or URL cannot be empty")

        if v.startswith(('http://', 'https://')):
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(v):
                raise ValueError("Invalid URL format")
        else:
            if len(v.strip()) < 1:
                raise ValueError("File path cannot be empty")
        return v.strip()

    @model_validator(mode='before')
    @classmethod
    def infer_file_source(cls, data):
        """Automatically infer file_source from file_path_or_url if not
         provided."""
        if isinstance(data, dict):
            file_source = data.get('file_source')
            file_path_or_url = data.get('file_path_or_url')

            # Only infer if file_source is not provided
            if file_source is None and file_path_or_url:
                if file_path_or_url.startswith(('http://', 'https://')):
                    data['file_source'] = FileSource.URL
                else:
                    data['file_source'] = FileSource.LOCAL_PATH

        return data

    @model_validator(mode='after')
    def validate_file_source_consistency(self):
        """Ensure file_source is consistent with file_path_or_url."""
        if self.file_source and self.file_path_or_url:
            is_url = self.file_path_or_url.startswith(('http://', 'https://'))
            expected_source = (FileSource.URL if is_url else
                               FileSource.LOCAL_PATH)

            if self.file_source != expected_source:
                raise ValueError(
                    f"File source '{self.file_source}' is inconsistent with "
                    f"file_path_or_url '{self.file_path_or_url}'. "
                    f"Expected '{expected_source}' for "
                    f"{'URL' if is_url else 'local path'}."
                )

        return self


class FileProcessingState(BaseModel):
    """State for the sub-graph that processes a single file."""
    original_path_or_url: Optional[str] = Field(
        default=None,
        description="Store the original input for context"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="This will be the actual local path used for processing"
    )
    file_type: Optional[str] = Field(
        default=None,
        description="Type/extension of the file (e.g. pdf, docx, etc.)"
    )
    extracted_text: Optional[str] = Field(
        default="",
        description="Text extracted from the file"
    )
    extracted_images: Optional[List[bytes]] = Field(
        default_factory=list, description="Stores raw image bytes"
    )

    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        """Validate that file type is not empty and is a valid extension."""
        if v is None:
            return v
        if not v or not v.strip():
            raise ValueError("File type cannot be empty")
        file_type = v.strip().lstrip('.').lower()
        if not file_type:
            raise ValueError("File type cannot be empty after processing")
        return file_type

    @field_validator('extracted_text')
    @classmethod
    def validate_extracted_text(cls, v):
        """Ensure extracted text is a string."""
        if v is None:
            return ""
        return str(v)

    @field_validator('extracted_images')
    @classmethod
    def validate_extracted_images(cls, v):
        """Ensure extracted images is a list of bytes."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("extracted_images must be a list")
        for img in v:
            if not isinstance(img, bytes):
                raise ValueError("All extracted images must be bytes")
        return v


class MainState(BaseModel):
    """The main state of the orchestration graph."""
    question: Optional[str] = Field(
        default=None,
        description="The user's question to be answered"
    )
    file_inputs: Optional[List[FileMetaData]] = Field(
        default_factory=list,
        description="List of file inputs to process"
    )
    processing_results: Optional[List[FileProcessingState]] = Field(
        default_factory=list,
        description="Results from processing each file"
    )
    aggregated_text: Optional[str] = Field(
        default="",
        description="Combined text from all processed files"
    )
    all_extracted_images: Optional[List[bytes]] = Field(
        default_factory=list,
        description="Stores raw image bytes from all files"
    )
    answer: Optional[str] = Field(
        default="",
        description="Final answer generated by the model"
    )

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate that the question is not empty."""
        if v is None:
            return v
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @field_validator('aggregated_text')
    @classmethod
    def validate_aggregated_text(cls, v):
        """Ensure aggregated text is a string."""
        if v is None:
            return ""
        return str(v)

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v):
        """Ensure answer is a string."""
        if v is None:
            return ""
        return str(v)

    @field_validator('all_extracted_images')
    @classmethod
    def validate_all_extracted_images(cls, v):
        """Ensure all_extracted_images is a list of bytes."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("all_extracted_images must be a list")
        for img in v:
            if not isinstance(img, bytes):
                raise ValueError("All extracted images must be bytes")
        return v

    def add_file_input(self, file_metadata: FileMetaData) -> None:
        """Add a file input to the state."""
        if self.file_inputs is None:
            self.file_inputs = []
        self.file_inputs.append(file_metadata)

    def add_processing_result(self, result: FileProcessingState) -> None:
        """Add a processing result to the state."""
        if self.processing_results is None:
            self.processing_results = []
        self.processing_results.append(result)

    def update_aggregated_text(self, text: str) -> None:
        """Update the aggregated text."""
        if text:
            if self.aggregated_text is None:
                self.aggregated_text = ""
            if self.aggregated_text:
                self.aggregated_text += "\n\n" + text
            else:
                self.aggregated_text = text

    def add_extracted_images(self, images: List[bytes]) -> None:
        """Add extracted images to the state."""
        if images:
            if self.all_extracted_images is None:
                self.all_extracted_images = []
            self.all_extracted_images.extend(images)

    def set_answer(self, answer: str) -> None:
        """Set the final answer."""
        self.answer = answer
