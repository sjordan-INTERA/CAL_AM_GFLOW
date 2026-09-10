from pathlib import Path

from PIL import Image, ImageDraw


PAGE_DIR = Path("tmp/pdfs/fig_5_11_scan")
PAGES_PER_SHEET = 20


def main():
    pages = sorted(PAGE_DIR.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No rendered PDF pages found in {PAGE_DIR}")

    # -- Combine rendered pages into compact sheets for visual review.
    for chunk_index in range(0, len(pages), PAGES_PER_SHEET):
        chunk = pages[chunk_index : chunk_index + PAGES_PER_SHEET]
        sheet = Image.new("RGB", (5 * 220, 4 * 280), "white")
        draw = ImageDraw.Draw(sheet)

        for page_index, page_path in enumerate(chunk):
            with Image.open(page_path) as source:
                page = source.convert("RGB")
            page.thumbnail((180, 240))

            x = (page_index % 5) * 220 + 20
            y = (page_index // 5) * 280 + 25
            sheet.paste(page, (x, y))
            draw.text((x, y - 18), page_path.name, fill="black")

        output_path = PAGE_DIR / f"contact_{chunk_index // PAGES_PER_SHEET + 1}.png"
        sheet.save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
