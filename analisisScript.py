import pandas as pd
from transformers import pipeline

# Lista de modelos que quieres probar
models = {

    # Modelos multilingües
    "nlptown": "nlptown/bert-base-multilingual-uncased-sentiment",
    "tabularisai": "tabularisai/multilingual-sentiment-analysis",

    # Modelos españoles
    "robertuito": "pysentimiento/robertuito-sentiment-analysis",
    "beto": "finiteautomata/beto-sentiment-analysis",
    "VP": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
    "edumunozsala": "edumunozsala/roberta_bne_sentiment_analysis_es",

    
}

# Frases difíciles de clasificar
texts = [
    "Mientras los franceses nos expoliaban con la excusa de hacer un museo nacional, los 'aliados' ingleses destruían nuestras industrias con la excusa de 'ayudarnos' a liberarnos.",
 "Hipócritas y mentirosos.",
 "Mucha hipocresía. Deberíamos aprender de los británicos y su respeto al patrimonio público y privado. Ellos lo conocen, lo preservan activamente con la colaboración ciudadana y respeto. No sirve de nada prohibir y medrar a costa de la hacienda pública sin proteger ni estudiar el patrimonio, dejándolo pudrirse. Leed la Constitución Española.",
 "En Inglaterra se descubren yacimientos nuevos todos los años, y no gracias a los arqueólogos, sino a los detectoristas, que están legalmente reconocidos. Ellos informan de los hallazgos y por eso se encuentra tanto: regulando, no prohibiendo.",
 "Debe de ser muy estresante vivir de subvenciones del Estado solo por mirar si hay un ladrillo o una teja en el suelo... y la mayoría de veces ni eso.",
 "La mayoría de cosas que se sacan con detector provienen de sitios totalmente descontextualizados, porque simplemente se cayeron ahí cuando pasó alguien, sin ningún contexto más.",
 "Tiene mucho valor que hablen estos señores que nunca trabajaron en su vida haciendo agujeros, mientras en yacimientos remueven dos metros de tierra que luego usan para tapar caminos a 10 km sin ni siquiera cribarla.",
 "La hipocresía es manifiesta. 'Solo nos interesa el contexto arqueológico', dicen. Pero en vez de proteger yacimientos, acumulan toneladas de monedas sin contexto que jamás serán conservadas ni estudiadas.",
 "Muy buenas. Me parece bien que defiendas tu pan, pero también debes respetar al coleccionista. Las monedas en la arqueología son el último eslabón. Hay toneladas en los museos que no se estudian. Bastarían unas pocas bien expuestas y vender el resto para financiar excavaciones. Hay que poner freno al mercado negro, sí, pero también permitir una afición que en muchos casos es de bajo nivel económico. No penséis solo en vuestro bienestar, también hay que respetar otras aficiones. Saludos.",
 "He visto el vídeo con retraso, pero opino que hay enfrentamiento entre arqueólogos y detectoristas. Entiendo que no debe profanarse un yacimiento, pero si encuentras una moneda en un sembrado no es delito. El problema es el comercio ilegal. Hay falta de regulación clara del Estado, y lo que necesitamos es que todos puedan convivir respetuosamente.",
 "Con todo respeto: ustedes son ambiguos. Solo cuentan lo que les interesa. Llevo 10 años con detector y nunca me llamaron 'pitero'. Si encuentro un cestercio en un sembrado de un amigo, ¿soy un criminal? Nunca vendí nada. Siempre doné hallazgos importantes a la administración. ¿Eso está mal también? Criminalizar a todos por igual es una oportunidad perdida para regular y cooperar con una actividad que no hace daño a nadie.",
 "Hay muchas situaciones así. En mi pueblo hay uno que va por yacimientos buscando. Las penas por expolio son muy bajas.",
 "Me encantó este directo. Es útil reflexionar sobre estos temas en este formato. Ojalá haya más en tu canal.",
 "Lo único discutible es afirmar que países como Francia, Italia o España son los más expoliados. Tienen gran parte de su patrimonio aún, y habría que compararlos con África, que sí sufrió expolio sistemático.",
 "Estimado Miguel Ángel, es un gusto escucharte. En la época napoleónica se robaron muchas obras, algunas devueltas. ¿Por qué Francia nunca devolvió 'Las bodas de Caná' a Italia?",
 "Muy buena tu idea sobre colaboraciones entre países. Soy mexicana y apoyo a nuestro presidente, pero si van a traer el Penacho, ¡que sea para conservarlo bien! Y que los europeos nos enseñen a hacerlo. Lo ideal sería que cada país tenga su patrimonio, pero la realidad es distinta.",
 "No compro el argumento de que los mármoles del Partenón tienen un contexto histórico en Londres. Cuando te roban, el ladrón debe demostrar que no lo ha hecho. No diste argumentos sólidos.",
 "Por favor, esas piezas son del Partenón, no del señor Elgin. Su único mérito fue arrancarlas de forma salvaje. Dejemos de llamarlas 'Mármoles Elgin'.",
 "La cuestión del penacho no es tanto si fue o no de Moctezuma, o si es original. El punto clave es la identidad cultural.",
 "El tema importante es la identidad. Hay muchas piezas en museos europeos que son claramente identitarias. Por eso deberían pedir perdón. Si fuésemos tan radicales como otros pueblos, esto ya habría estallado.",
 "¿Se tiene en cuenta que si no las hubieran conservado estudiosos occidentales muchas piezas se habrían perdido? Además, muchas fueron vendidas por sus propios dueños por unas monedas.",
 "No es justo comparar el expolio británico con tener a la Dama de Elche o de Baza en el Arqueológico, accesibles a todos.",
 "No es mi opinión porque DIOS NO QUIERE IMÁGENES MATERIALES.",
"Las monedas se consiguen, nada de lo que es propio de la Policía Nacional es robado. Pero el Estado y la Policía las están expropiando para sus propios intereses.",
 "Si Francia aprobara una ley de devolución, España sería de las más beneficiadas por la cantidad de obras expoliadas en el extranjero.",
 "Las cosas son de su sitio y nada más.",
 "El arte es de quien lo encuentra y se lo queda. Y si hay que defenderlo del gobierno, pues se defiende.",
 "Excelente video.",
 "Esa es la Dama de Elche. Nadie sabe quién la hizo ni su historia. ¿Ustedes sí? Saludos desde Ciudad de México.",
 "No creo que la Tizona o la Colada del Cid en Madrid sean las originales, pero pertenecieron a figuras importantes de la Corona de Aragón.",
 "El patrimonio robado a España por los Borbones y los franceses, como los códices mayas del Escorial o los restos románicos en NY, debería ser devuelto.",
 "Aunque el mayor expolio lo cometieron los propios españoles durante la Guerra Civil. El bando rojo destruyó incontables obras de arte, pero de eso no se habla.",
 "Sería bueno que España devuelva los quipus que confiscó. Así podríamos conocer la escritura de los incas.",
 "No estoy de acuerdo con dispersar las piezas de los grandes museos españoles. Las colecciones deben mantenerse unidas.",
 "Qué curioso que los museos más visitados del mundo tengan más del 50% de sus piezas robadas.",
 "Está claro: las piezas saqueadas por ingleses y franceses deben volver. Pero si lo hicieran, el Louvre y el British quedarían medio vacíos. Y no olvidemos lo que Napoleón llevó a Francia, eso también debe devolverse.",
 "En Latinoamérica muchos se quejan del imperio español, pero no ven que los ingleses eran peores: racistas, saqueadores y esclavistas.",
 "El MAN y el Prado son patrimonio de todos los españoles. Deben preservarse.",
 "No hay problema en que esté en Elche o en Madrid. El MAN es de todos. Igual que el Guernica, que si sale, debería ir a Málaga. Si no, que se quede en la capital.",
 "Qué coraje. ¡El patrimonio español debe volver a España! Y a los guiris, que les den. ¡Viva España!",
 "Me llama la atención que ningún país le reclame nada a España. Por ejemplo, hay cuadros de Velázquez en Reino Unido que podríamos recuperar.",
 "Y en España está el Museo del Jamón, que debería devolverse a Extremadura.",
 "El arte es... morirse de frío 😂🤣😂",
 "¿Estarían igual de conservadas si hubieran permanecido en su lugar de origen? En España hay muchos edificios históricos en ruinas.",
 "La mejor solución sería que los países expoliados recibieran un porcentaje de la recaudación de los museos como 'alquiler'.",
 "La mafia del gobierno también saca su parte.",
 "Sí, que regresen todo. Que México recupere lo que le han robado.",
 "Stop al expolio arqueológico. No debería permitirse la compraventa de bienes culturales. Solo cesiones temporales en situaciones de emergencia y con devolución posterior.",
 "Estaría bien que España devolviera lo que robó a Latinoamérica... y que aún sigue robando. Nos robaron hasta la conciencia.",
 "Al final los que deberían rendir cuentas son las monjas por vender lo que el Estado había apropiado. Y subsidiariamente, la Iglesia."

]


# Cargar modelos
classifiers = {}
for name, model_id in models.items():
    try:
        classifiers[name] = pipeline("sentiment-analysis", model=model_id)
    except Exception as e:
        print(f"Error cargando modelo {name}: {e}")
        classifiers[name] = None

# Preparar tabla con predicciones
rows = []

for text in texts:
    row = {"Comentario": text}
    for name, classifier in classifiers.items():
        if classifier is None:
            row[name] = "ERROR"
        else:
            try:
                result = classifier(text)[0]
                label = result["label"]
                score = result["score"]
                row[name] = f"{label} ({score:.2f})"
            except Exception as e:
                row[name] = f"ERROR: {str(e)}"
    rows.append(row)

# Exportar a CSV
df = pd.DataFrame(rows)
df.to_csv("resultados_por_modelo.csv", index=False, encoding="utf-8")