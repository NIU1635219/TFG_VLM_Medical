\# Limitaciones Técnicas y Bugs Cognitivos en VLMs aplicados a Imagen Médica



Durante la ejecución de la Fase 2 del estudio de ablación (n=2500 inferencias), se monitorizó el comportamiento de modelos SOTA (Qwen3.5, MiniCPM-o) bajo condiciones de estrés. El análisis empírico reveló diversas limitaciones arquitectónicas intrínsecas que comprometen el rendimiento autónomo de los VLMs en entornos clínicos.



\## 1. Discretización de Parches y Cascada de Errores en la Regresión de Bounding Boxes

A pesar de demostrar una excelente atención visual (evaluada mediante \*Proximity Score\*), los modelos siguen sin ser suficientemente precisos al intentar indicar, mediante la generación de coordenadas (\*Bounding Boxes\*), la ubicación exacta de las lesiones clínicas. Esta imprecisión geométrica se debe a tres fallos arquitectónicos fundamentales de la visión computacional basada en Transformers:



\*   \*\*Procesamiento por Parches (Patchification):\*\* Los modelos no procesan la imagen píxel a píxel, sino que la dividen en fragmentos o \*patches\* (típicamente de 14x14 o 16x16 píxeles). El modelo aproxima el borde de la lesión al límite del \*patch\* más cercano, lo que impide, por diseño, una metrología o regresión de bordes milimétrica.

\*   \*\*Cuantización Espacial y Error en Cascada:\*\* La precisión se reduce drásticamente en imágenes de alta resolución porque el modelo debe dividir un espacio continuo en una cuadrícula discreta de 1000 puntos (escala normalizada `\[0-1000]`). En consecuencia, el modelo solo genera posiciones relativas aproximadas. Además, dada la naturaleza autoregresiva de la generación de texto (token a token), un ligero error de cuantización espacial al predecir la primera coordenada induce una \*\*cascada de fallos geométricos\*\* en los puntos sucesivos, deformando irremediablemente la caja respecto a los bordes reales del objeto.

\*   \*\*Desacoplamiento Arquitectónico (Módulos Independientes vs. Entrenados en la Misma Capa):\*\* Durante el estudio, se evidenció una profunda "agnosia espacial" en arquitecturas como \*\*MiniCPM-o\*\*. Al utilizar un codificador visual y un modelo de lenguaje que no han sido entrenados al mismo tiempo (módulos acoplados a posteriori), el modelo es capaz de entender los elementos de la imagen general, pero fracasa al intentar situarlos físicamente en el lienzo. 

Por este motivo, el proyecto se centró casi en exclusiva en la familia \*\*Qwen3.5\*\*, cuya arquitectura sí entrena tanto la visión como el texto en las mismas capas simultáneamente (\*Early Fusion\* o Fusión Temprana nativa). Sin embargo, se demostró que, aunque Qwen3.5 reduce drásticamente la agnosia espacial frente a MiniCPM-o, ni siquiera el entrenamiento en la misma capa logra superar los problemas subyacentes de la \*patchification\* y la cuantización previamente descritos.



\## 2. El Bucle de Alucinación Recursiva y la Entropía de Generación

Durante la experimentación, se observó que los modelos entraban frecuentemente en bucles de autorreflexión destructiva (ej. cadenas recurrentes del tipo: \*"wait... the texture is smooth... but wait, the borders are indistinct... wait..."\*). Inicialmente, este fenómeno se atribuyó en exclusiva al uso del \*Thinking Mode\*, pero el análisis del flujo de inferencia reveló un defecto inherente a las arquitecturas generativas: \*\*la incapacidad nativa para determinar el final óptimo de la inferencia\*\*.



Este error sistémico se fundamenta en la naturaleza estocástica y autoregresiva de los modelos:

\*   \*\*Existencia Token a Token:\*\* Los modelos carecen de metacognición fuera del proceso de generación; "solo existen" en el acto de emitir el siguiente token. No pueden pre-planificar la longitud exacta o la conclusión de su respuesta clínica.

\*   \*\*Exceso de Libertad (`max\_tokens` elevado):\*\* Al configurar un límite de tokens amplio en la API, se le otorga al modelo "demasiado espacio vacío" que, probabilísticamente, siente la necesidad de rellenar. 

\*   \*\*Contaminación de Contexto (Autosugestión):\*\* Al prolongar artificialmente la generación, el propio texto generado pasa a formar parte de su ventana de contexto de forma inmediata. El modelo termina "leyéndose a sí mismo", confundiéndose con sus propias hipótesis no concluyentes y cayendo en un bucle entrópico del que es incapaz de salir.



\## 3. Violación de Contrato Estructural por Saturación de Prompts

En el Escenario E (máxima asistencia contextual: BBox rojo + Ground Truth inyectado + esquema JSON de Pydantic estricto), se detectó una sorpresiva caída en la fiabilidad del formato de salida. 



El modelo sufrió un colapso de atención (\*Attention Window Saturation\*). Al exigirle procesar simultáneamente una instrucción de sistema (\*System Prompt\*) compleja, alteraciones artificiales en la imagen (líneas rojas), y obligarle a cumplir con un esquema rígido de campos estructurados, el modelo incurrió repetidamente en violaciones de contrato: ignoró la clase médica inyectada en el prompt o alteró la estructura de las llaves del JSON esperado. Esto demuestra empíricamente el techo de carga cognitiva en el \*prompting\* multimodal.



\## 4. El Impacto de la Sintaxis Nativa y la Alineación de Esquemas

Durante la experimentación espacial con \*\*Qwen3.5\*\*, se evidenció la importancia crítica de respetar las estructuras de datos con las que el modelo fue entrenado originalmente. 



Al corregir el esquema Pydantic para solicitar las coordenadas en el orden nativo de entrenamiento de Qwen (`\[xmin, ymin, xmax, ymax]`) en lugar de otras variantes, el número de errores de validación geométrica \*\*se redujo drásticamente, casi desapareciendo\*\*. 

Esto subraya un principio fundamental en MLOps: forzar al modelo a utilizar estructuras de datos que entran en conflicto con la topología de su entrenamiento original genera una disonancia cognitiva profunda. Esta discrepancia no solo provoca violaciones de formato, sino que empeora severamente el rendimiento analítico general y la eficiencia de la inferencia al confundir las capas de decodificación.



\## 5. La Degradación del Espacio Latente Visual por Cuantización (Q4 vs Q8)

Uno de los descubrimientos más relevantes de la experimentación local fue el impacto asimétrico de la cuantización espacial sobre las modalidades de visión y texto. 



Mientras que en los LLMs de texto puro bajar a 4 bits (Q4) produce una degradación marginal, en los VLMs médicos el paradigma cambia radicalmente:

\*   \*\*Colapso de Características de Alta Frecuencia:\*\* Los detalles endoscópicos sutiles (microcapilares) se representan como variaciones matemáticas minúsculas. Al aplicar una cuantización agresiva, estos vectores precisos se agrupan en escalones enteros rígidos, provocando una pérdida masiva de resolución matemática en el espacio latente.

\*   \*\*Miopía Inducida:\*\* Este "pixelado matemático interno" provoca que el modelo sufra de miopía clínica, siendo incapaz de diferenciar entre patologías de texturas finas, a pesar de que el módulo lingüístico mantiene intacta su capacidad de redactar textos gramaticalmente perfectos. 



\## 6. Impacto del Ruido Lumínico en la Atención Visual

Las imágenes endoscópicas presentan frecuentemente reflejos especulares intensos causados por el flash del propio endoscopio. Los VLMs son altamente sensibles a esta saturación lumínica. En múltiples casos, el mecanismo de atención del modelo confundió los brillos especulares blancos con "capas de moco" o "exudado fibrinoso" (características clínicas de HP o ASS), evidenciando una grave vulnerabilidad a los artefactos de iluminación que induce a falsos positivos.



\---



\## Conclusión: La Inevitabilidad de la Alucinación Estocástica

A pesar de aplicar metodologías avanzadas de inyección de contexto restrictivo (\*Pipeline Asistido\*) y formatos estrictos de salida (\*Structured Outputs\*), la investigación concluye que \*\*es matemáticamente imposible garantizar al 100% que el modelo no alucine o que acate la instrucción de trabajar exclusivamente con el diagnóstico proporcionado\*\*. 



Esta limitación es inherente a la arquitectura de los Modelos Fundacionales actuales:

1\.  \*\*Naturaleza Probabilística No Determinista:\*\* Al ser sistemas estocásticos, siempre subsiste un margen donde el modelo priorizará una secuencia de tokens estadísticamente frecuente en sus pesos de entrenamiento, por encima de la instrucción restrictiva temporal proporcionada en el \*prompt\*.

2\.  \*\*Fragmentación de la Atención Limitada:\*\* El mecanismo de atención (\*Attention Mechanism\*) dispone de una capacidad computacional finita. En este TFG, la atención debe dividirse forzosamente entre miles de tokens visuales de la imagen y múltiples fuentes textuales que añaden tokens (System Prompt, validadores de Pydantic, Ground Truth inyectado). Cuando esta atención se comparte y satura entre tantas fuentes diversas, el modelo pierde el anclaje de la directiva principal y recurre a inferencias probabilísticas generales descontroladas.



En definitiva, las arquitecturas SOTA de 2026 demuestran una capacidad de razonamiento latente sin precedentes, pero su ineludible comportamiento probabilístico confirma que el despliegue de sistemas de explicabilidad clínica basados en VLMs requiere supervisión humana obligatoria (\*Human-in-the-Loop\*) por diseño, y no por un defecto de implementación del software.

