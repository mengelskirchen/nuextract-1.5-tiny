"""Aufruf: python nuextract15_tiny.py dokument.pdf felder.json"""
import json
import sys

import pymupdf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELLNAME = "numind/NuExtract-1.5-tiny"
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 1024


def main():
    if len(sys.argv) != 3:
        sys.exit("Aufruf: python nuextract15_tiny.py dokument.pdf felder.json")

    # Ausschließlich vorhandenen Text von Seite 1 lesen, keine OCR.
    with pymupdf.open(sys.argv[1]) as pdf:
        if len(pdf) == 0:
            sys.exit("Das PDF enthält keine Seiten.")
        text = pdf[0].get_text("text")
    if not text.strip():
        sys.exit("Seite 1 enthält keinen auslesbaren Text. Zuerst OCR durchführen.")

    with open(sys.argv[2], encoding="utf-8-sig") as f:
        felder = json.load(f)
    if not isinstance(felder, dict) or not felder:
        sys.exit("Die JSON-Vorlage muss ein nicht leeres Objekt mit Feldnamen sein.")

    # NuExtract 1.5 verwendet eine Beispielstruktur mit leeren Werten.
    # Anders als bei NuExtract3 NICHT durch den Typnamen "string" ersetzen.
    vorlage = json.dumps(felder, ensure_ascii=False, indent=4)
    prompt = (
        "<|input|>\n### Template:\n" + vorlage
        + "\n### Text:\n" + text + "\n\n<|output|>"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELLNAME)
    eingabe = tokenizer(prompt, return_tensors="pt", truncation=False)
    tokenzahl = eingabe.input_ids.shape[1]
    if tokenzahl > MAX_INPUT_TOKENS:
        sys.exit(
            f"Die Eingabe hat {tokenzahl} Tokens; dieses Beispiel erlaubt höchstens "
            f"{MAX_INPUT_TOKENS}. Text kürzen oder eine Chunk-Verarbeitung ergänzen."
        )

    # Ohne GPU: FP32 auf CPU. Keine CUDA-Erweiterungen oder torchvision nötig.
    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    datentyp = torch.float16 if geraet == "cuda" else torch.float32
    print(f"Lade {MODELLNAME} auf {geraet} ...", file=sys.stderr)
    modell = AutoModelForCausalLM.from_pretrained(
        MODELLNAME, dtype=datentyp
    ).to(geraet).eval()
    eingabe = eingabe.to(geraet)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        ausgabe = modell.generate(
            **eingabe,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_id,
        )

    # Nur neu erzeugte Tokens dekodieren, nicht den Eingabeprompt.
    antwort = tokenizer.decode(
        ausgabe[0, tokenzahl:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).split("<|end-output|>", 1)[0].strip()

    try:
        daten = json.loads(antwort)
    except json.JSONDecodeError:
        sys.exit(
            "Das Modell hat kein gültiges JSON erzeugt. "
            "Möglicherweise wurde das Ausgabelimit erreicht. Modellausgabe:\n"
            + antwort[:2000]
        )
    if not isinstance(daten, dict):
        sys.exit("Das Modell hat kein JSON-Objekt erzeugt. Modellausgabe:\n" + antwort[:2000])
    print(json.dumps(daten, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
