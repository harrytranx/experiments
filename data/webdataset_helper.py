# pyre-strict
import json
import logging
import os
import tarfile
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

from PIL import Image
import io
from tqdm import tqdm

DEFAULT_SHARD_SUFFIX_ID_LEN = 8
DEFAULT_SAMPLE_SUFFIX_ID_LEN = 5

def num_to_str_id(num: int, str_len: int) -> str:
    """
    Convert a number to a string ID with a specified length.
    Args:
        num (int): The number to convert.
        str_len (int): The desired length of the resulting string ID.
    Returns:
        A string ID with the specified length.
    """
    str_num = str(num)
    str_id = "0" * (str_len - len(str_num)) + str_num
    return str_id


class Webdataset:
    def __init__(
        self, json_data: List[Dict[str, Any]], image_handles: List[io.BytesIO]
    ) -> None:
        self.json_data = json_data
        self.image_handles = image_handles

    def to_buffer(self, **kwargs: Any) -> Tuple[Union[BytesIO, str], Dict[str, Any]]:
        """
        Write to in-memory buffer
        """
        return self.create_shard(
            json_data=self.json_data,
            image_handles=self.image_handles,
            output="buffer",
            **kwargs,
        )

    def to_file(
        self, output_tar_file: str, **kwargs: Any
    ) -> Tuple[Union[BytesIO, str], Dict[str, Any]]:
        """
        Write to .tar file
        """
        return self.create_shard(
            json_data=self.json_data,
            image_handles=self.image_handles,
            output=output_tar_file,
            **kwargs,
        )

    def create_shard(
        self,
        json_data: List[Dict[str, Any]],
        image_handles: List[io.BytesIO],
        sample_prefix: str = "",
        sample_suffix_id_len: int = DEFAULT_SAMPLE_SUFFIX_ID_LEN,
        output: str = "buffer",
        progress_bar: bool = True,
    ) -> Tuple[Union[str, BytesIO], Dict[str, Any]]:
        """
        output: "buffer", or "path/to/file.tar"
        Return output file name (*.tar)
        """

        if output == "buffer":  # write to in-memory buffer
            tar_buffer = BytesIO()
            tarfile_args = {"fileobj": tar_buffer, "mode": "w"}
        elif ".tar" in output:  # write to file
            shard_file = output
            if os.path.exists(shard_file):
                os.remove(shard_file)

            shard_file_tmp = shard_file.replace(".tar", ".tmp")
            tarfile_args = {"name": shard_file_tmp, "mode": "w"}
        else:
            raise ValueError("output must be 'buffer' or 'path/to/file.tar'")
        
        image_objects = image_handles

        num_samples = len(json_data)
        valid_samples = 0
        with tarfile.open(**tarfile_args) as tar:  # pyre-ignore[6]
            iterator = range(num_samples)
            if progress_bar:
                iterator = tqdm(range(num_samples))

            for i in iterator:
                sample_id = num_to_str_id(i, sample_suffix_id_len)
                sample_name = f"{sample_prefix}{sample_id}"

                try:
                    image_buffer = image_objects[i]

                    if image_buffer is None:
                        logging.debug(
                            f"Image buffer at index {i} is invalid"
                        )
                        continue

                    self._save_json_to_tar(json_data[i], tar, f"{sample_name}.json")

                    self._save_image_buffer_to_tar(image_buffer, tar, f"{sample_name}.jpg")

                    valid_samples += 1
                except Exception:
                    logging.debug(
                        f"An error occurred when trying to fetch image with everstore handle {image_handles[i]}"
                    )

        stats = {"provided_samples": num_samples, "valid_samples": valid_samples}

        if output == "buffer":
            tar_buffer.seek(0)
            return tar_buffer, stats  # pyre-ignore[61]
        else:
            os.rename(shard_file_tmp, shard_file)  # pyre-ignore[61]
            return shard_file, stats  # pyre-ignore[61]

    def _save_json_to_tar(
        self, json_dict: Dict[str, Any], tar_file: tarfile.TarFile, json_file_name: str
    ) -> None:
        """
        Save a dict as a .json file inside a tar archive
        json_file_name must include .json extension
        """

        # Write the dictionary to an in-memory file-like object
        buffer = BytesIO()
        buffer.write(json.dumps(json_dict, indent=4).encode())

        # Seek to the beginning of the file-like object
        buffer.seek(0)

        # Add the in-memory file-like object to the tar archive
        info = tarfile.TarInfo(name=json_file_name)
        info.size = len(buffer.getvalue())
        tar_file.addfile(info, fileobj=buffer)

    def _save_image_buffer_to_tar(
        self,
        image_buffer: Image.Image,
        tar_file: tarfile.TarFile,
        image_file_name: str,
        image_format: str = "JPEG",
    ) -> None:
        """
        Save an image object to tar achive
        """

        info = tarfile.TarInfo(name=image_file_name)

        # # Create an in-memory file-like object to store the image data
        # buffer = BytesIO()
        # pil_image.save(buffer, format=image_format)

        # Get the size of the image data in bytes
        image_buffer.seek(0)
        data = image_buffer.getbuffer()
        info.size = len(data)

        # Add the in-memory file-like object to the tar archive
        image_buffer.seek(0)
        tar_file.addfile(info, fileobj=image_buffer)
        
        
    def _save_image_to_tar(
        self,
        pil_image: Image.Image,
        tar_file: tarfile.TarFile,
        image_file_name: str,
        image_format: str = "JPEG",
    ) -> None:
        """
        Save an image object to tar achive
        """

        info = tarfile.TarInfo(name=image_file_name)

        # Create an in-memory file-like object to store the image data
        buffer = BytesIO()
        pil_image.save(buffer, format=image_format)

        # Get the size of the image data in bytes
        buffer.seek(0)
        data = buffer.getbuffer()
        info.size = len(data)

        # Add the in-memory file-like object to the tar archive
        buffer.seek(0)
        tar_file.addfile(info, fileobj=buffer)

    def _save_buffer_to_tar(
        self, buffer: BytesIO, tar_file: tarfile.TarFile, buffer_file_name: str
    ) -> None:
        """
        Save BytesIO buffer to a file inside a tar archive
        """
        info = tarfile.TarInfo(name=buffer_file_name)
        info.size = len(buffer.getbuffer())
        tar_file.addfile(info, fileobj=buffer)
