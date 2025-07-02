import mimetypes
import os
import requests
import tempfile

from src.utils.logger import get_logger

# Initialize logger at module level
logger = get_logger("download_helper")


def download_file_from_url(url: str) -> str:
    """Downloads a file from a URL to a temporary local file."""
    logger.info("Attempting to download file from URL: %s", url)

    try:
        response = requests.get(url, stream=True, timeout=30)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        # Get file extension from URL if possible, or default to .bin
        content_type = response.headers.get('Content-Type', '')
        extension = mimetypes.guess_extension(content_type)
        if not extension:
            # Try to infer from URL path
            path = requests.utils.urlparse(url).path
            ext = os.path.splitext(path)[-1]
            if ext:
                extension = ext
            else:
                extension = ".bin"

        # Create a temporary file to store the downloaded content
        # Use delete=False so we can manually delete it later in a
        # try-finally block
        temp_file = tempfile.NamedTemporaryFile(delete=False,
                                                suffix=extension)
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        temp_file.close()
        logger.info("Successfully downloaded %s to %s", url, temp_file.name)
        return temp_file.name
    except requests.exceptions.RequestException as e:
        logger.error("Error downloading file from %s: %s", url, e)
        raise ValueError(f"Failed to download file from URL: {url} - {e}")
    except Exception as e:
        logger.error("An unexpected error occurred during download from " +
                     f"{url}: {e}")
        raise Exception("An unexpected error occurred during download " +
                        f"from {url}: {e}")
