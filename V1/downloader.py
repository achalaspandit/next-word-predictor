import requests
import time
import random
import os


def download_text_file(url="https://www.gutenberg.org/files/1661/1661-0.txt", 
                      output_filename="1661-0.txt", 
                      max_retries=5, 
                      retry_delay_seconds=15,
                      force_redownload=False):
    
    if not force_redownload and os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping download.")
        return True
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/117.0 Safari/537.36",
    ]
    
    BASE_HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    download_successful = False

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} of {max_retries} to download {output_filename}...")
        try:
            headers = BASE_HEADERS.copy()
            headers["User-Agent"] = random.choice(USER_AGENTS)

            with requests.get(url, headers=headers, timeout=10, stream=True) as response:
                response.raise_for_status()
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 8192
                with open(output_filename, 'wb') as file:
                    downloaded_bytes = 0
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            file.write(chunk)
                            downloaded_bytes += len(chunk)
                            
            print(f"\nSuccessfully downloaded {output_filename}!")
            download_successful = True
            break
            
        except requests.exceptions.RequestException as e:
            print(f"Error on attempt {attempt}: An unexpected request error occurred: {e}")
        except IOError as e:
            print(f"Error on attempt {attempt}: Could not write file {output_filename}: {e}")

        if attempt < max_retries:
            print(f"Retrying in {retry_delay_seconds}s...")
            time.sleep(retry_delay_seconds)
        else:
            print(f"Max retries ({max_retries}) reached. Failed to download {output_filename}.")

    if download_successful:
        print(f"\File '{output_filename}' is available in your working directory.")
        return True
    else:
        print("\nFile download failed after all attempts.")
        return False


def download_sherlock_holmes_text(force_redownload=False):
    """
    Download the Sherlock Holmes text specifically.
    
    Args:
        force_redownload (bool): Force redownload even if file exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    return download_text_file(
        url="https://www.gutenberg.org/files/1661/1661-0.txt",
        output_filename="1661-0.txt",
        force_redownload=force_redownload
    )


if __name__ == "__main__":
    download_sherlock_holmes_text()