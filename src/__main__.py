import argparse
import json
import os
from typing import Any
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model  # type: ignore
from . import FunctionsDefinition, FunctionCallingTests
from .ai_core import ai_name, ai_parameters


def main() -> None:
    """
    Punto di ingresso principale dell'applicazione.
    Gestisce gli argomenti da riga di comando, carica i dati JSON di input
    e le definizioni delle funzioni,
    esegue l'LLM per estrarre nomi e parametri,
    e infine salva i risultati formattati in un file di output.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Il mio strumento AI per Function Calling")

        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
            help="Il file con le regole delle funzioni")
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="Il file con le domande")
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
            help="Il file dove salveremo i risultati")

        args = parser.parse_args()

        lista_funzioni: list[FunctionsDefinition] = []
        lista_prompt: list[FunctionCallingTests] = []

        try:
            with open(args.functions_definition, "r") as file:
                dati_funzioni = json.load(file)
            for x in dati_funzioni:
                functionvalida = FunctionsDefinition(**x)
                lista_funzioni.append(functionvalida)

            with open(args.input, "r") as file:
                dati_prompt = json.load(file)
            for x in dati_prompt:
                promptvalido = FunctionCallingTests(**x)
                lista_prompt.append(promptvalido)

        except FileNotFoundError as e:
            print(f"error: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"error: {e}")
            return
        except ValidationError as e:
            for error in e.errors():
                print(error['msg'])
            return

        ai = Small_LLM_Model()
        risultati_finali: list[dict[str, Any]] = []

        for prompt_obj in lista_prompt:
            name = ai_name(lista_funzioni, ai, prompt_obj)
            parametri_estratti_da_ai = ai_parameters(
                lista_funzioni, ai, prompt_obj, name)
            risposta_singola = {
                "prompt": prompt_obj.prompt,
                "name": name,
                "parameters": parametri_estratti_da_ai
            }
            risultati_finali.append(risposta_singola)

        cartella_output = os.path.dirname(args.output)

        if cartella_output:
            os.makedirs(cartella_output, exist_ok=True)

        with open(args.output, "w") as file:
            json.dump(risultati_finali, file, indent=4)
    except Exception as e:
        print(f"Errore inaspettato durante l'esecuzione: {e}")


if __name__ == "__main__":
    main()
