import io
import logging
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import (
    DoclingDocument,
    TableItem,
    PictureItem,
    TextItem,
    DocItem
)
from models.chunk import (
    Chunk as ChunkData,
    ChunkType
)
from models.file import FileSource

logger = logging.getLogger(__name__)

TOKENIZER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENIZER_TOKENS = 256
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

async def Chunk(file_source : FileSource):
    chunker = _create_chunker()
    converted_info = await _convert_to_docling_file(file_source)
    docling_file = converted_info.document

    # get table chunks
    table_chunks = _get_table_chunks(docling_file=docling_file)
    # get image chunks
    image_chunks = _get_image_chunks(docling_file=docling_file)
    # get text chunks
    text_chunks = _get_text_chunks(chunker=chunker, docling_file=docling_file)

    all_chunks = list(ChunkData)
    all_chunks.extend(table_chunks)
    all_chunks.extend(image_chunks)
    all_chunks.extend(text_chunks)



def _create_chunker() -> HybridChunker:
    # check if a tokenizer is needed if we are not actually embedding the chunk text
    chunker = HybridChunker(
        tokenizer=TOKENIZER_MODEL_NAME,
        max_tokens=MAX_TOKENIZER_TOKENS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, 
        merge_peers=False,
    )

    return chunker

async def _convert_to_docling_file(file_source : FileSource) -> ConversionResult:
    converter = DocumentConverter()

    pdf_stream = io.BytesIO(file_source.content)
    try:
        doc_stream = DocumentStream(name=file_source.name, stream=pdf_stream)
        result = converter.convert(doc_stream)
        return result
    finally:
        pdf_stream.close()

def _get_text_chunks(chunker : HybridChunker, docling_file : DoclingDocument) -> list[ChunkData]:
    # remove the items which are not text items from docling document before chunking to keep text chunks purely of text items
    try:
        non_text_items =  non_text_items = [
            item
            for item in docling_file.iterate_items()
            if not isinstance(item, TextItem)
        ]
        docling_file.delete_items(node_items=non_text_items)
    except Exception as e:
        logger.error("_get_text_chunks : error deleting non text items", extra={"error": str(e)})
        raise

    try:
        chunks = []
        for chunk in chunker.chunk(dl_doc=docling_file):
            # find page number and bbox
            page_no = 1
            bbox = None
            meta = chunk.meta
            if hasattr(meta,"doc_items") and meta.doc_items:
                doc_item = meta.doc_items[0]
                if hasattr(doc_item,"prov") and doc_item.prov:
                    page_no = doc_item.prov.page_no 

                    bbox_obj = doc_item.prov.bbox
                    if all(hasattr(bbox_obj, attr) for attr in ["l", "t", "r", "b"]):
                        bbox = [
                            float(bbox_obj.l),
                            float(bbox_obj.t),
                            float(bbox_obj.r),
                            float(bbox_obj.b),
                        ]

            text = getattr(chunk, "text", None)

            chunk = ChunkData(
                chunk_type=ChunkType.TEXT,
                text_content=text,
                page_no=page_no,
                bbox=bbox
            )
            chunks.append(chunk)

        return chunks
    except Exception as e:
        logger.error("_get_text_chunks : error extracting text chunks", extra={"error" : str(e)})
        raise
        

def _get_table_chunks(docling_file : DoclingDocument) -> list[ChunkData]:
    try:
        chunks = list()
        for item in docling_file.iterate_items():
            if isinstance(item, TableItem):
                # Get page number, defaults to 1 if no page number is found
                page_no = 1
                if item.prov:
                    prov = item.prov[0]
                    page_no = prov.page_no

                # Get bounding box data
                bbox = _get_item_bbox(item=item)

                table_data = _extract_table_markdown_from_item(item, docling_file)
                if not table_data:
                    continue

                chunk = ChunkData(
                    chunk_type=ChunkType.TABLE,
                    table_content_markdown=table_data,
                    page_no=page_no,
                    bbox=bbox
                )

                chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error("_get_table_chunks : error getting table markdown", extra={"error": str(e)})
        raise

def _extract_table_markdown_from_item(item : TableItem, docling_file : DoclingDocument)-> str:
    """
        Extract table data in markdown representation from TableItem
    """
    try:
        if hasattr(item, "export_to_dataframe"):
            df = item.export_to_dataframe(doc=docling_file)
            if df.empty:
                return None

            df=df.fillna("")

            markdown = df.to_markdown(index=False, tablefmt="github")
            return markdown
        else:
            logger.warning("_extract_table_markdown_from_item : TableItem has no method export_to_dataframe. Skipping extraction of table data")
            return None
    except Exception as e:
            logger.error("_extract_table_markdown_from_item : error extracting table markdown", extra={"error": str(e)})
            raise

def _get_image_chunks(docling_file : DoclingDocument) -> list[ChunkData]:
    try:
        chunks = list()
        for item in docling_file.iterate_items():
            if isinstance(item, PictureItem):
                # Get page number, defaults to 1 if no page number is found
                page_no = 1
                if item.prov:
                    prov = item.prov[0]
                    page_no = prov.page_no

                # Get bounding box data
                bbox = _get_item_bbox(item=item)

                image = _extract_image_from_item(item=item,docling_file=docling_file)
                if not image:
                    continue

                chunk = ChunkData(
                    chunk_type=ChunkType.IMAGE,
                    image_content=image,
                    page_no=page_no,
                    bbox=bbox
                )

                chunks.append(chunk)
        return chunks
    except Exception as e:
            logger.error("_get_image_chunks : error getting image", extra={"error": str(e)})
            raise

def _extract_image_from_item(item: PictureItem, docling_file : DoclingDocument) -> Optional[Image.Image]:
        """
        Extract PIL Image from PictureItem.
        """
        try:
            if hasattr(item, "get_image"):
                return item.get_image(doc=docling_file)
            else:
                logger.warning("PictureItem has no get_image method. Skipping extraction of image")
                return None
        except Exception as e:
            logger.error("_extract_image_from_item : error extracting image", extra={"error": str(e)})
            raise

def _get_item_bbox(self, item: DocItem) -> list[float]:
        if hasattr(item, "prov") and len(item.prov) > 0:
            prov = item.prov[0]
            if hasattr(prov, "bbox"):
                bbox = prov.bbox
                if all(hasattr(bbox, attr) for attr in ["l", "t", "r", "b"]):
                    # Return as [left, top, right, bottom]
                    return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]

        return None




                
             
        