import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from . import FunctionCallingTests


def extrat_string(
    ai: Small_LLM_Model,
    prompt_obj: FunctionCallingTests,
    nome_parametro: str,
    diz_da_riempire: dict[str, Any]
) -> str:
    """
    Estrae il valore esatto di una stringa dal testo generato dall'LLM.
    Utilizza un prompt strutturato per isolare un parametro specifico e
    scorre i token generati finché non incontra un ritorno a capo ('\\n').
    """
    frase_prompt = (
        "Rule: Copy the exact value from the text. Do not remove \
         slashes (/) and do not remove quotes \n\n"
        "Text: \"Read the file at /var/log/sys.log\"\n"
        "Extract the parameter:\n"
        "path: /var/log/sys.log\n\n"
        "Text: \"Format template: Say \\\"hi\\\" to {name}\"\n"
        "Extract the parameter:\n"
        "template: Say \"hi\" to {name}\n\n"
        f"Text: \"{prompt_obj.prompt}\"\n"
        "Extract the parameter:\n"
    )

    for chiave_salvata, valore_salvato in diz_da_riempire.items():
        frase_prompt += f"{chiave_salvata}: {valore_salvato}\n"

    frase_prompt += f"{nome_parametro}: "

    frase_prompt_ids: list[int] = ai.encode(frase_prompt)[0].tolist()
    risposta_generata: list[int] = []

    while True:
        logits = ai.get_logits_from_input_ids(frase_prompt_ids)

        nuovo_token = int(np.argmax(logits))
        risposta_generata.append(nuovo_token)
        frase_prompt_ids.append(nuovo_token)

        ultimo_char = ai.decode([nuovo_token])

        if '\n' in ultimo_char:
            break
    return str(ai.decode(risposta_generata))


def extrat_num(
    ai: Small_LLM_Model,
    prompt_obj: FunctionCallingTests,
    nome_parametro: str,
    diz_da_riempire: dict[str, Any],
    dizionario_ai: dict[str, int],
    tipo_dato: str
) -> str:
    """
    Estrae un valore numerico generato dal modello LLM
      garantendo il formato corretto.
    Filtra i token permessi (solo cifre, segni, spazi e
    punti decimali se necessario)
    per forzare l'AI a rispondere esclusivamente con un numero valido.
    """
    frase_prompt = (
        "Rule: Extract the exact numbers from the text in order."
        "Do not repeat the same number for different parameters.\n\n"
        "Text: 'What is the product of 15 and 2?'\n"
        "Extract the numbers:\n"
        "a: 15\n"
        "b: 2\n\n"
        f"Text: '{prompt_obj.prompt}'\n"
        "Extract the numbers:\n"
    )

    for chiave_salvata, valore_salvato in diz_da_riempire.items():
        frase_prompt += f"{chiave_salvata}: {valore_salvato}\n"
    frase_prompt += f"{nome_parametro}: "

    frase_prompt_ids: list[int] = ai.encode(frase_prompt)[0].tolist()
    risposta_generata: list[int] = []
    token_permessi: list[int] = []

    for token_testo, token_id in dizionario_ai.items():
        testo_pulito = token_testo.strip("Ġ ▂▃▄▅▆▇█")
        if testo_pulito.isdigit():
            token_permessi.append(token_id)

    token_permessi.append(ai.encode("-")[0].tolist()[-1])
    token_permessi.append(ai.encode(" ")[0].tolist()[-1])
    token_permessi.append(ai.encode("\n")[0].tolist()[-1])
    if tipo_dato == "number":
        token_permessi.append(ai.encode(".")[0].tolist()[-1])

    while True:
        logits = ai.get_logits_from_input_ids(frase_prompt_ids)
        if len(token_permessi) > 0:
            for x in range(len(logits)):
                if x not in token_permessi:
                    logits[x] = float('-inf')

        nuovo_token = int(np.argmax(logits))
        risposta_generata.append(nuovo_token)
        frase_prompt_ids.append(nuovo_token)

        ultimo_char = ai.decode([risposta_generata[-1]])
        if ultimo_char in [" ", "\n"] and len(risposta_generata) > 0:
            break

    return str(ai.decode(risposta_generata))
