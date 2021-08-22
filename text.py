
from typing import Dict


tipo_de_frase = ["pregunta", "orden"]

verbos = dict()
prefijos = dict()
sufijos = dict()

verbos["orden"] = ["reanuda", "continúa",]
verbos["pregunta"] = ["reanudar", "continuar con"]
verbos["expresion"] = ["reanudes", "continues"]

prefijos["pregunta"] = ["puedes", "oye puedes", "podrías", "tienes chance de"]
prefijos["orden"] = ["oye", ""]
prefijos["expresion"] = ["ocupo que", "necesito que"]


sufijos["orden"] = ["porfa", "por favor", ""]
sufijos["pregunta"] = sufijos["orden"]
sufijos["expresion"] = sufijos["orden"]

sujetos = ["la impresión", "la impresión 3D", "lo que se está imprimiendo", "lo que tengo en la impresora", "la pieza que se esta imprimieno"]

for tipo in tipo_de_frase:
        for verbo in verbos[tipo]:
            for sufijo in sufijos[tipo]:
                for prefijo in prefijos[tipo]:
                    for sujeto in sujetos:
                        print((prefijo + " " + verbo + " " + sujeto + " " + sufijo).strip())