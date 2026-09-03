import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from . import FunctionsDefinition, FunctionCallingTests, SchemaTypeInfo
from .ai_extractors import extrat_string, extrat_num


def ai_parameters(
    lista_funzioni: list[FunctionsDefinition],
    ai: Small_LLM_Model,
    prompt_obj: FunctionCallingTests,
    name: str
) -> dict[str, Any]:
    """
    Estrae e formatta i parametri richiesti per una specifica funzione.
    Legge lo schema della funzione individuata, controlla
    il tipo di dato di ogni
    parametro (string, number, integer) e delega
    l'estrazione alla funzione appropriata.
    """
    parametri: dict[str, SchemaTypeInfo] | None = None
    percorso_vocab = ai.get_path_to_vocab_file()

    with open(percorso_vocab, "r") as file:
        dizionario_ai = json.load(file)

    for funzione in lista_funzioni:
        if funzione.name == name:
            parametri = funzione.parameters
            break

    diz_da_riempire: dict[str, Any] = {}

    if parametri is None:
        return diz_da_riempire

    for chiave, valore in parametri.items():
        nome_parametro = chiave
        tipo_dato = valore.type

        if tipo_dato in ("number", "integer"):
            risultato_num = extrat_num(ai, prompt_obj, nome_parametro,
                                       diz_da_riempire,
                                       dizionario_ai, tipo_dato)
            risultato_num = risultato_num.strip()
            if tipo_dato == "number":
                diz_da_riempire[nome_parametro] = float(risultato_num)
            else:
                diz_da_riempire[nome_parametro] = int(risultato_num)

        elif tipo_dato == "string":
            risultato_str = extrat_string(ai, prompt_obj,
                                          nome_parametro,
                                          diz_da_riempire)
            risultato_pulito = risultato_str.replace(
                                    '\n', '').strip().strip('"')
            diz_da_riempire[nome_parametro] = risultato_pulito

    return diz_da_riempire


def ai_name(
    lista_funzioni: list[FunctionsDefinition],
    ai: Small_LLM_Model,
    prompt_obj: FunctionCallingTests
) -> str:
    """
    Identifica il nome della funzione corretta da invocare in base al prompt.
    Genera un prompt a scelta multipla e forza l'AI a rispondere solo con
    l'indice numerico corrispondente alla funzione più adatta.
    """
    frase_prompt = ("Select the exact function number for the given text.\n"
                    "\nAvailable functions:\n")

    for indice, funzione in enumerate(lista_funzioni):
        frase_prompt += f"[{indice}] {funzione.name}: {funzione.description}\n"

    frase_prompt += f"\nText: '{prompt_obj.prompt}'\nFunction number: "

    percorso_vocab = ai.get_path_to_vocab_file()
    with open(percorso_vocab, "r") as file:
        dizionario_ai = json.load(file)

    token_permessi: list[int] = []
    for token_testo, token_id in dizionario_ai.items():
        testo_pulito = token_testo.strip("Ġ ▂▃▄▅▆▇█")
        if testo_pulito.isdigit():
            token_permessi.append(token_id)

    token_permessi.append(ai.encode(" ")[0].tolist()[-1])
    token_permessi.append(ai.encode("\n")[0].tolist()[-1])

    prompt_ids: list[int] = ai.encode(frase_prompt)[0].tolist()
    risposta_generata: list[int] = []

    while True:
        logits = ai.get_logits_from_input_ids(prompt_ids)
        print(logits)
        for x in range(len(logits)):
            if x not in token_permessi:
                logits[x] = float('-inf')
        nuovo_token = int(np.argmax(logits))
        ultimo_char = ai.decode([nuovo_token])
        if ultimo_char in [" ", "\n"]:
            break
        risposta_generata.append(nuovo_token)
        prompt_ids.append(nuovo_token)

    stringa_numero = ai.decode(risposta_generata).strip()

    if not stringa_numero.isdigit():
        stringa_numero = "0"
    indice_scelto = int(stringa_numero)

    if indice_scelto >= len(lista_funzioni):
        indice_scelto = 0

    nome_funzione_scelta = lista_funzioni[indice_scelto].name

    return nome_funzione_scelta
