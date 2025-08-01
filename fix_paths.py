import os
import re
import urllib.parse
from glob import glob

def find_md_file(category, date_str):
    search_path = os.path.join(category, "_posts", f"{date_str}-*.md")
    matches = glob(search_path)
    if not matches:
        raise FileNotFoundError(f"No markdown file found under {search_path}")
    if len(matches) > 1:
        print("[경고] 여러 개의 md 파일이 발견되었습니다. 첫 번째 파일만 처리합니다.")
    return matches[0]

def convert_image_paths(md_path, category, date_str):
    asset_prefix = f"/assets/img/{category}/{date_str}/"

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        matches = re.findall(r'!\[(.*?)\]\(([^)]+)\)', line)
        for alt_text, raw_filename in matches:
            decoded_filename = urllib.parse.unquote(raw_filename)
            if '/' not in raw_filename:
                new_path = f"{asset_prefix}{decoded_filename}"
                line = line.replace(f"]({raw_filename})", f"]({new_path})")
        new_lines.append(line)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[✔] 이미지 경로 업데이트 완료: {md_path}")

def main():
    category = input("카테고리 이름을 입력하세요 (예: 블로그): ").strip()
    date_str = input("날짜를 입력하세요 (YYYY-MM-DD 형식): ").strip()

    try:
        md_file = find_md_file(category, date_str)
        convert_image_paths(md_file, category, date_str)
    except Exception as e:
        print(f"[오류] {e}")

if __name__ == "__main__":
    main()
