"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""


from __future__ import annotations
from typing import Any

print("--------------------------------------------------")
print("!!! CHÍNH XÁC LÀ FILE BẠN SỬA ĐANG ĐƯỢC CHẠY !!!")
print("--------------------------------------------------")

PROMPTS: dict[str, Any] = {}

# System prompts for different analysis types
# PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
#     "Bạn là một chuyên gia phân tích hình ảnh. Hãy cung cấp mô tả chi tiết, chính xác bằng tiếng Việt."
# )
# PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
#     "You are an expert image analyst. Provide detailed analysis based on available information."
# )
# PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
#     "You are an expert data analyst. Provide detailed table analysis with specific insights."
# )
# PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
#     "You are an expert mathematician. Provide detailed mathematical analysis."
# )
# PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
#     "You are an expert content analyst specializing in {content_type} content."
# )

# System prompts for different analysis types
# System prompt dung de dinh nghia vai tro ngu canh truoc khi dua vao yeu cau phan tich cu the
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích hình ảnh. Hãy cung cấp mô tả chi tiết, chính xác bằng tiếng Việt."
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích hình ảnh. Hãy cung cấp phân tích chi tiết dựa trên thông tin có sẵn bằng tiếng Việt."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích dữ liệu. Hãy cung cấp phân tích bảng biểu chi tiết với các thông tin cụ thể bằng tiếng Việt."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "Bạn là một chuyên gia toán học. Hãy cung cấp phân tích toán học chi tiết bằng tiếng Việt."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích nội dung chuyên về loại nội dung {content_type}. Hãy phân tích chi tiết bằng tiếng Việt."
)

# Image analysis prompt template
# Image analysis prompt template
PROMPTS[
    "vision_prompt"
] = """Hãy phân tích hình ảnh này một cách chi tiết và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một mô tả trực quan toàn diện và chi tiết về hình ảnh theo các hướng dẫn sau:
    - Trả lời bằng Tiếng Việt.
    - Mô tả bố cục và cách sắp xếp tổng thể.
    - Xác định tất cả các đối tượng, con người, văn bản và các yếu tố trực quan.
    - Giải thích mối quan hệ giữa các yếu tố.
    - Ghi chú về màu sắc, ánh sáng và phong cách hình ảnh.
    - Mô tả bất kỳ hành động hoặc hoạt động nào được hiển thị.
    - Bao gồm chi tiết kỹ thuật nếu có liên quan (biểu đồ, sơ đồ, v.v.).
    - Luôn sử dụng tên cụ thể thay vì đại từ.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "tóm tắt ngắn gọn nội dung hình ảnh và ý nghĩa của nó bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Thông tin bổ sung:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Tập trung vào việc cung cấp phân tích trực quan chính xác, chi tiết, hữu ích cho việc truy xuất kiến thức."""

# Image analysis prompt with context support
# CAC PROMPT WITH CONTEXT SUPPORT CHO PHEP DUA THEM NGU CANH XUNG QUANH VAO PROMPT DE PHAN TICH CHINH XAC HON
# KHI HINH ANH DAU VAO XUNG QUANH CO TEXT HOAC ANH KHÔNG CUNG CẤP DỦ THÔNG TIN ĐỂ XÁC ĐỊNH NỘI DUNG THÌ SẼ SƯ DỤNG PROMPT _WITH_CONTEXT

PROMPTS[
    "vision_prompt_with_context"
] = """Hãy phân tích hình ảnh này một cách chi tiết, có xem xét đến bối cảnh xung quanh. Cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một mô tả trực quan toàn diện và chi tiết về hình ảnh theo các hướng dẫn sau:
    - Trả lời bằng Tiếng Việt.
    - Mô tả bố cục và cách sắp xếp tổng thể.
    - Xác định tất cả các đối tượng, con người, văn bản và các yếu tố trực quan.
    - Giải thích mối quan hệ giữa các yếu tố và cách chúng liên quan đến bối cảnh xung quanh.
    - Ghi chú về màu sắc, ánh sáng và phong cách hình ảnh.
    - Mô tả bất kỳ hành động hoặc hoạt động nào được hiển thị.
    - Bao gồm chi tiết kỹ thuật nếu có liên quan (biểu đồ, sơ đồ, v.v.).
    - Tham chiếu các kết nối đến nội dung xung quanh khi có liên quan.
    - Luôn sử dụng tên cụ thể thay vì đại từ.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "tóm tắt ngắn gọn nội dung hình ảnh, ý nghĩa và mối quan hệ với nội dung xung quanh bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Ngữ cảnh từ nội dung xung quanh:
{context}

Chi tiết hình ảnh:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Tập trung vào việc cung cấp phân tích trực quan chính xác, chi tiết, kết hợp ngữ cảnh và hữu ích cho việc truy xuất kiến thức."""

# Image analysis prompt with text fallback
PROMPTS["text_prompt"] = """Dựa trên thông tin hình ảnh sau đây, hãy cung cấp phân tích bằng Tiếng Việt:

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}"""

# Table analysis prompt template
PROMPTS[
    "table_prompt"
] = """Please analyze this table content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }}
}}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data."""

# Table analysis prompt with context support
# Table analysis prompt template
PROMPTS[
    "table_prompt"
] = """Hãy phân tích nội dung bảng này và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về bảng bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Cấu trúc và cách tổ chức của bảng
    - Tiêu đề cột và ý nghĩa của chúng
    - Các điểm dữ liệu chính và các mẫu (patterns)
    - Thông tin chi tiết về thống kê và xu hướng
    - Mối quan hệ giữa các yếu tố dữ liệu
    - Ý nghĩa của dữ liệu được trình bày
    - Luôn sử dụng tên và giá trị cụ thể thay vì tham chiếu chung chung.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "tóm tắt ngắn gọn mục đích của bảng và các phát hiện chính bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Thông tin bảng:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Tập trung vào việc trích xuất những thông tin chi tiết và mối quan hệ có ý nghĩa từ dữ liệu bảng."""

# Table analysis prompt with context support
PROMPTS[
    "table_prompt_with_context"
] = """Hãy phân tích nội dung bảng này, có xem xét đến bối cảnh xung quanh, và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về bảng bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Cấu trúc và cách tổ chức của bảng
    - Tiêu đề cột và ý nghĩa của chúng
    - Các điểm dữ liệu chính và các mẫu (patterns)
    - Thông tin chi tiết về thống kê và xu hướng
    - Mối quan hệ giữa các yếu tố dữ liệu
    - Ý nghĩa của dữ liệu được trình bày trong mối liên hệ với bối cảnh xung quanh
    - Cách bảng hỗ trợ hoặc minh họa các khái niệm từ nội dung xung quanh
    - Luôn sử dụng tên và giá trị cụ thể thay vì tham chiếu chung chung.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "tóm tắt ngắn gọn mục đích của bảng, các phát hiện chính và mối quan hệ với nội dung xung quanh bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Ngữ cảnh từ nội dung xung quanh:
{context}

Thông tin bảng:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Tập trung vào việc trích xuất những thông tin chi tiết và mối quan hệ có ý nghĩa từ dữ liệu bảng trong bối cảnh nội dung xung quanh."""

# Equation analysis prompt template
# Equation analysis prompt template
PROMPTS[
    "equation_prompt"
] = """Hãy phân tích phương trình toán học này và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về phương trình bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Ý nghĩa và diễn giải toán học.
    - Các biến số và định nghĩa của chúng.
    - Các phép toán và hàm được sử dụng.
    - Lĩnh vực ứng dụng và ngữ cảnh vật lý hoặc lý thuyết.
    - Mối quan hệ với các khái niệm toán học khác.
    - Các ứng dụng thực tế hoặc trường hợp sử dụng.
    - Luôn sử dụng thuật ngữ toán học cụ thể.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "tóm tắt ngắn gọn mục đích và ý nghĩa của phương trình bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Thông tin phương trình:
Equation: {equation_text}
Format: {equation_format}

Tập trung vào việc cung cấp những hiểu biết toán học sâu sắc và giải thích ý nghĩa của phương trình."""

# Equation analysis prompt with context support
PROMPTS[
    "equation_prompt_with_context"
] = """Hãy phân tích phương trình toán học này, có xem xét đến bối cảnh xung quanh, và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về phương trình bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Ý nghĩa và diễn giải toán học.
    - Các biến số và định nghĩa của chúng trong bối cảnh nội dung xung quanh.
    - Các phép toán và hàm được sử dụng.
    - Lĩnh vực ứng dụng và ngữ cảnh dựa trên tài liệu xung quanh.
    - Ý nghĩa vật lý hoặc lý thuyết.
    - Mối quan hệ với các khái niệm toán học khác được đề cập trong ngữ cảnh.
    - Các ứng dụng thực tế hoặc trường hợp sử dụng.
    - Cách phương trình liên quan đến cuộc thảo luận hoặc khung lý thuyết rộng hơn.
    - Luôn sử dụng thuật ngữ toán học cụ thể.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "tóm tắt ngắn gọn mục đích, ý nghĩa và vai trò của phương trình trong bối cảnh xung quanh bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Ngữ cảnh từ nội dung xung quanh:
{context}

Thông tin phương trình:
Equation: {equation_text}
Format: {equation_format}

Tập trung vào việc cung cấp những hiểu biết toán học sâu sắc và giải thích ý nghĩa của phương trình trong bối cảnh rộng hơn."""

# Generic content analysis prompt template
# Generic content analysis prompt template
PROMPTS[
    "generic_prompt"
] = """Hãy phân tích nội dung loại {content_type} này và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về nội dung bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Cấu trúc và cách tổ chức nội dung.
    - Thông tin và các yếu tố chính.
    - Mối quan hệ giữa các thành phần.
    - Bối cảnh và ý nghĩa.
    - Các chi tiết liên quan cho việc truy xuất kiến thức.
    - Luôn sử dụng thuật ngữ cụ thể phù hợp với nội dung loại {content_type}.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "tóm tắt ngắn gọn mục đích và các điểm chính của nội dung bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Content: {content}

Tập trung vào việc trích xuất thông tin có ý nghĩa hữu ích cho việc truy xuất kiến thức."""

# Generic content analysis prompt with context support
PROMPTS[
    "generic_prompt_with_context"
] = """Hãy phân tích nội dung loại {content_type} này, có xem xét đến bối cảnh xung quanh, và cung cấp phản hồi JSON theo cấu trúc sau:

{{
    "detailed_description": "Một phân tích toàn diện về nội dung bao gồm:
    - Trả lời bằng Tiếng Việt.
    - Cấu trúc và cách tổ chức nội dung.
    - Thông tin và các yếu tố chính.
    - Mối quan hệ giữa các thành phần.
    - Bối cảnh và ý nghĩa trong mối liên hệ với nội dung xung quanh.
    - Cách nội dung này kết nối hoặc hỗ trợ cuộc thảo luận rộng hơn.
    - Các chi tiết liên quan cho việc truy xuất kiến thức.
    - Luôn sử dụng thuật ngữ cụ thể phù hợp với nội dung loại {content_type}.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "tóm tắt ngắn gọn mục đích, các điểm chính và mối quan hệ với nội dung xung quanh bằng Tiếng Việt (tối đa 100 từ)"
    }}
}}

Ngữ cảnh từ nội dung xung quanh:
{context}

Content: {content}

Tập trung vào việc trích xuất thông tin có ý nghĩa hữu ích cho việc truy xuất kiến thức và hiểu vai trò của nội dung trong bối cảnh rộng hơn."""

# Modal chunk templates
# DÙNG ĐỂ CHUYỂN CÁC KẾT QUẢ JSON SAU KHI PHÂN TÍCH XONG ĐƯỢC CHUYỂN THÀNH CÁC VĂN BẢN TEXT ĐỂ LƯU VÀO DB
PROMPTS["image_chunk"] = """
Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""

# Query-related prompts
# DÙNG ĐỂ HƯỚNG DẪN AI CÁCH TỔNG HƠP THÔNG TIN TÌM ĐƯỢC ĐỂ TRẢ LỜI CHO USER
# HỆ THỐNG SẼ NHÌN VÀO DỮ LIỆU GỬI KÈM ĐỂ CHUẨN BỊ CÁC LOẠI PROMPT TRƯỚC RỒI SỬ DỤNG
# LLM ĐỂ PHÂN TÍCH CÂU HỎI CỦA USER VÀ ĐỌC LẠI KẾT QUẢ TỪ CÁC KẾT QUẢ ĐÃ ĐƯỢC CHUẨN BỊ 
# RỒI MỚI IN RA KẾT QUẢ PHÙ HỢP VỚI CÂU HỎI


PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Hãy mô tả ngắn gọn nội dung chính, các yếu tố quan trọng và thông tin nổi bật trong hình ảnh này bằng Tiếng Việt."
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích hình ảnh chuyên nghiệp, có khả năng mô tả chính xác nội dung hình ảnh bằng Tiếng Việt."
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """Hãy phân tích nội dung chính, cấu trúc và thông tin quan trọng của dữ liệu bảng sau:

Dữ liệu bảng:
{table_data}

Chú thích bảng: {table_caption}

Hãy tóm tắt ngắn gọn nội dung chính, đặc điểm dữ liệu và các phát hiện quan trọng bằng Tiếng Việt."""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích dữ liệu chuyên nghiệp, người có thể phân tích chính xác dữ liệu bảng biểu bằng Tiếng Việt."
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """Hãy giải thích ý nghĩa và mục đích của công thức toán học sau:

Công thức LaTeX: {latex}
Chú thích: {equation_caption}

Hãy giải thích ngắn gọn ý nghĩa toán học, kịch bản ứng dụng và tầm quan trọng của công thức này bằng Tiếng Việt."""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "Bạn là một chuyên gia toán học, người có thể giải thích rõ ràng các công thức toán học bằng Tiếng Việt."
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """Hãy phân tích nội dung loại {content_type} sau đây và trích xuất thông tin chính:

Nội dung: {content_str}

Hãy tóm tắt ngắn gọn các đặc điểm chính và thông tin quan trọng bằng Tiếng Việt."""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "Bạn là một chuyên gia phân tích nội dung chuyên nghiệp, người có thể phân tích chính xác loại nội dung {content_type} bằng Tiếng Việt."
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\n**QUAN TRỌNG - QUY TẮC TRẢ LỜI:**\n"
    "1. CHỈ sử dụng thông tin có trong context được cung cấp ở trên.\n"
    "2. KHÔNG bịa đặt, suy đoán hoặc thêm thông tin từ kiến thức riêng.\n"
    "3. Nếu không tìm thấy thông tin trong context, hãy trả lời: 'Không tìm thấy thông tin này trong tài liệu.'\n"
    "4. Trích dẫn nguồn thông tin khi có thể.\n\n"
    "Vui lòng cung cấp câu trả lời toàn diện dựa trên câu hỏi của người dùng và thông tin nội dung đa phương thức được cung cấp. "
    "Hãy trả lời hoàn toàn bằng Tiếng Việt một cách tự nhiên và chuyên nghiệp."
)