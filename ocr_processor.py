import cv2
import numpy as np
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans

class OCRProcessor:
    def __init__(self, lang='ru'):
        # В версии 2.7.3 параметр show_log РАБОТАЕТ!
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False  # ← РАБОТАЕТ в 2.7.3!
        )
    
    def get_side_centers(self, polygon):
        pts = np.array(polygon)
        sorted_pts = pts[pts[:, 0].argsort()]
        left_pts = sorted_pts[:2]
        right_pts = sorted_pts[-2:]
        return np.mean(left_pts, axis=0), np.mean(right_pts, axis=0)
    
    def build_lines_by_local_neighbors(self, words, max_vertical_diff=20, max_horizontal_gap=150):
        if len(words) <= 1:
            return [words] if words else []
        
        word_data = []
        for word in words:
            left_c, right_c = self.get_side_centers(word['polygon'])
            word_data.append({
                'word': word,
                'left_center': left_c,
                'right_center': right_c,
                'used': False
            })
        
        lines = []
        for start in word_data:
            if start['used']:
                continue
            current_line = [start]
            start['used'] = True
            current = start
            while True:
                candidates = []
                for cand in word_data:
                    if cand['used']:
                        continue
                    if cand['left_center'][0] > current['right_center'][0]:
                        vertical_diff = abs(current['right_center'][1] - cand['left_center'][1])
                        horizontal_gap = cand['left_center'][0] - current['right_center'][0]
                        if vertical_diff <= max_vertical_diff and horizontal_gap <= max_horizontal_gap:
                            candidates.append((cand, horizontal_gap))
                if not candidates:
                    break
                candidates.sort(key=lambda x: x[1])
                next_word = candidates[0][0]
                next_word['used'] = True
                current = next_word
                current_line.append(current)
            lines.append(current_line)
        
        lines.sort(key=lambda line: min(w['left_center'][1] for w in line))
        return [[w['word'] for w in line] for line in lines]
    
    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        H, W = img.shape[:2]
        
        # В версии 2.7.3 параметр cls=True РАБОТАЕТ!
        result = self.ocr.ocr(img, cls=True)  # ← РАБОТАЕТ в 2.7.3!
        
        if result[0] is None:
            return {"blocks": []}
        
        word_blocks = []
        for item in result[0]:
            polygon = item[0]
            text = item[1][0]      # ← В 2.7.3 это кортеж (text, confidence)
            confidence = item[1][1]
            
            pts = np.array(polygon)
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            
            word_blocks.append({
                'polygon': polygon,
                'bbox': [x1, y1, x2, y2],
                'text': text,
                'confidence': float(confidence)
            })
        
        if not word_blocks:
            return {"blocks": []}
        
        # ... остальной код (как в Colab) ...
        bboxes = np.array([w['bbox'] for w in word_blocks])
        centers_x = np.array([(b[0] + b[2]) / 2 for b in bboxes]).reshape(-1, 1)
        x_std = np.std(centers_x) if len(centers_x) > 1 else 0
        n_columns = 2 if (x_std > W * 0.2 and len(word_blocks) > 5) else 1
        
        if n_columns > 1:
            kmeans = KMeans(n_clusters=n_columns, n_init=10)
            labels = kmeans.fit_predict(centers_x)
            columns = []
            for i in range(n_columns):
                col_words = [word_blocks[j] for j in range(len(word_blocks)) if labels[j] == i]
                if col_words:
                    columns.append(col_words)
            columns.sort(key=lambda col: np.mean([w['bbox'][0] for w in col]))
        else:
            columns = [word_blocks]
        
        blocks = []
        block_id = 1
        for col_idx, col_words in enumerate(columns):
            lines = self.build_lines_by_local_neighbors(col_words)
            for line in lines:
                if not line:
                    continue
                line_text = " ".join([w['text'] for w in line])
                all_pts = np.vstack([np.array(w['polygon']) for w in line])
                x1 = int(all_pts[:, 0].min())
                y1 = int(all_pts[:, 1].min())
                x2 = int(all_pts[:, 0].max())
                y2 = int(all_pts[:, 1].max())
                
                blocks.append({
                    "block_id": block_id,
                    "type": "line",
                    "bbox": [x1, y1, x2, y2],
                    "text": line_text,
                    "column_id": col_idx + 1,
                    "word_count": len(line)
                })
                block_id += 1
        
        return {"blocks": blocks}