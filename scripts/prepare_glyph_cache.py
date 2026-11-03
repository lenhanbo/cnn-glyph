import argparse

from models.feature_query import save_glyph_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unifont", default="files/unifont.pickle")
    parser.add_argument("--output", default="files/gnu-32x32.pt")
    parser.add_argument("--metadata", default="files/gnu-32x32.json")
    args = parser.parse_args()
    save_glyph_cache(args.unifont, args.output, args.metadata)
