from downloader import download_sherlock_holmes_text
from data_processor import prepare_data_and_embeddings


if __name__ == "__main__":
    download_sherlock_holmes_text()
    prepare_data_and_embeddings()

    