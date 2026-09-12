from deep_translator import GoogleTranslator


def translator(text: str) -> str:
    translator = GoogleTranslator(source="auto", target="en")

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    translated_chunks = []

    for chunk in chunks:
        translated_chunks.append(translator.translate(chunk))

    return " ".join(translated_chunks)