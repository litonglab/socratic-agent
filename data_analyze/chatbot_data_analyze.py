import pdfplumber
import re
import json
import os
import time

class PDFStructureParser:
    def __init__(self, pdf_path, output_image_folder):
        self.pdf_path = pdf_path
        # 获取文件名（不含扩展名），用于生成图片前缀
        self.pdf_filename = os.path.basename(pdf_path).replace('.pdf', '')
        self.output_image_folder = output_image_folder
        self.data = [] 

    def is_inside_bbox(self, obj_bbox, container_bboxes):
        """判断对象是否在容器（如表格）内部"""
        x0, top, x1, bottom = obj_bbox
        cx, cy = (x0 + x1) / 2, (top + bottom) / 2
        for box in container_bboxes:
            bx0, btop, bx1, bbottom = box
            if bx0 <= cx <= bx1 and btop <= cy <= bbottom:
                return True
        return False

    def get_heading_level(self, text):
        """正则判断标题层级"""
        pattern = r"^\s*(\d+(\.\d+)*)\.?\s+(.*)"
        match = re.match(pattern, text)
        if match:
            numbering = match.group(1)
            title = match.group(3)
            level = numbering.count('.') + 1
            return level, numbering, title
        return None

    def parse(self):
        """执行解析"""
        with pdfplumber.open(self.pdf_path) as pdf:
            all_content_stream = []

            for page_idx, page in enumerate(pdf.pages):
                current_page_num = page_idx + 1
                
                # --- 1. 提取表格 (Tag: t) ---
                tables = page.find_tables()
                table_bboxes = [t.bbox for t in tables]
                
                for table in tables:
                    all_content_stream.append({
                        "tag": "t",
                        "page": current_page_num,
                        "top": table.bbox[1],
                        "bbox": table.bbox, # 保留表格坐标
                        "content": table.extract()
                    })

                # --- 2. 提取并保存图片 (Tag: p) ---
                for img_idx, img in enumerate(page.images):
                    # 获取坐标 (x0, top, x1, bottom)
                    img_bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                    
                    # 过滤过小的图标/线条
                    if (img['bottom'] - img['top']) < 10 or (img['x1'] - img['x0']) < 10: 
                        continue

                    try:
                        # --- 截图保存逻辑 ---
                        cropped = page.crop(img_bbox, strict=False)
                        im_obj = cropped.to_image(resolution=300)
                        
                        # 生成相对路径文件名
                        img_filename = f"{self.pdf_filename}_p{current_page_num}_{img_idx}.png"
                        # 拼接完整保存路径
                        img_save_path = os.path.join(self.output_image_folder, img_filename)
                        
                        im_obj.save(img_save_path)

                        # --- 数据录入 ---
                        all_content_stream.append({
                            "tag": "p",
                            "page": current_page_num,
                            "top": img['top'],
                            # 1. 这里保留了原始 PDF 中的坐标 (用于前端高亮)
                            "bbox": img_bbox,  
                            # 2. 这里保留了切片后的 PNG 文件路径 (用于RAG展示/多模态识别)
                            "image_path": img_save_path, 
                            "width": img['width'],
                            "height": img['height']
                        })
                    except Exception as e:
                        print(f"警告: 图片提取失败 (页 {current_page_num}), 原因: {e}")

                # --- 3. 提取文字 (Tag: w) ---
                words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
                lines = []
                current_line = []
                last_bottom = 0
                
                for word in words:
                    word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                    if self.is_inside_bbox(word_bbox, table_bboxes):
                        continue
                    if current_line and (word['top'] - last_bottom) > 5:
                        lines.append(current_line)
                        current_line = []
                    current_line.append(word)
                    last_bottom = word['top']
                
                if current_line: lines.append(current_line)

                for line in lines:
                    if not line: continue
                    text_str = " ".join([w['text'] for w in line])
                    all_content_stream.append({
                        "tag": "w",
                        "page": current_page_num,
                        "top": min([w['top'] for w in line]),
                        "text": text_str,
                        "font_size": line[0].get('size', 0)
                    })

            # --- 4. 排序与层级构建 ---
            all_content_stream.sort(key=lambda x: (x['page'], x['top']))

            root = {"level": 0, "children": []}
            node_stack = [root]

            for item in all_content_stream:
                if item['tag'] == 'w':
                    heading_info = self.get_heading_level(item['text'])
                    if heading_info:
                        level, number, title = heading_info
                        new_section = {
                            "type": "section",
                            "number": number,
                            "title": title,
                            "level": level,
                            "children": []
                        }
                        while len(node_stack) > 1 and node_stack[-1].get('level', 0) >= level:
                            node_stack.pop()
                        node_stack[-1]['children'].append(new_section)
                        node_stack.append(new_section)
                    else:
                        node_stack[-1]['children'].append(item)
                else:
                    node_stack[-1]['children'].append(item)

            return root['children']

def process_folder(input_folder, output_json_file):
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 '{input_folder}' 不存在")
        return

    # 创建保存图片的文件夹
    images_dir = "extracted_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # 确保图片文件夹路径是规范的相对路径字符串
    # 这样 JSON 里的 path 也是相对路径，方便迁移
    images_dir_clean = os.path.normpath(images_dir)

    all_results = []
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    
    print(f"开始处理 {total_files} 个文件...")

    for idx, filename in enumerate(pdf_files):
        file_path = os.path.join(input_folder, filename)
        print(f"[{idx+1}/{total_files}] 处理中: {filename} ...")
        
        try:
            parser = PDFStructureParser(file_path, images_dir_clean)
            structured_data = parser.parse()
            
            all_results.append({
                "file_name": filename,
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "structure_data": structured_data
            })
            
        except Exception as e:
            print(f"  -> 失败: {filename}, 错误: {str(e)}")

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n全部完成！")
    print(f"JSON数据: {output_json_file}")
    print(f"图片目录: {images_dir_clean}/")

# ================= 配置区 =================
if __name__ == "__main__":
    # 你的PDF文件夹路径
    INPUT_FOLDER = "./labs_pdf" 
    # 输出的JSON文件名
    OUTPUT_JSON = "all_labs_structured.json"
    
    process_folder(INPUT_FOLDER, OUTPUT_JSON)