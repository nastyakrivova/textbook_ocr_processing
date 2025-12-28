import argparse
import json
import os
from ocr_processor import OCRProcessor

def main():
    parser = argparse.ArgumentParser(description="OCR для фото учебников")
    parser.add_argument("image_path", help="Путь к изображению")
    parser.add_argument("-o", "--output", default="output.json", help="Путь к JSON-файлу (по умолчанию: output.json)")
    parser.add_argument("--lang", default="ru", help="Язык OCR (по умолчанию: ru)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл {args.image_path} не найден")
        return
    
    print(f"Обработка изображения: {args.image_path}")
    processor = OCRProcessor(lang=args.lang)
    result = processor.process_image(args.image_path)
    
    # Сохранение в JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Результат сохранён в: {args.output}")
    print(f"Найдено блоков: {len(result['blocks'])}")

if __name__ == "__main__":
    main()
