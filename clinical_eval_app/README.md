# Clinical Eval App

Aplicacion web estatica para validacion clinica de justificaciones de IA (Match / No Match).

## Objetivo

Permitir que evaluadores medicos revisen casos de forma local (sin backend), registren veredictos y exporten resultados en CSV para analisis posterior.

## Estructura esperada

- index.html
- style.css
- app.js
- data/cases.json
- data/images/

## Esquema de entrada (cases.json)

Cada caso debe seguir este formato:

```json
[
  {
    "id_imagen": "polyp_042",
    "image_path": "data/images/polyp_042_bbox.jpg",
    "ground_truth_class": "AD",
    "id_modelo": "qwen3.5-9b",
    "clinical_justification": "La lesion presenta un patron vascular irregular..."
  }
]
```

Notas:

- id_modelo no se muestra en la UI (doble ciego), pero se usa en el CSV exportado.
- image_path debe ser una ruta relativa valida desde index.html.

## Funcionamiento

1. La app carga data/cases.json.
2. Muestra un caso cada vez con su imagen, diagnostico real y justificacion.
3. El evaluador vota Match o No Match y puede anadir comentario opcional.
4. Se puede navegar con flechas (anterior/siguiente), incluida entrada manual a la pagina final desde el ultimo caso con la flecha de avance.
5. Se guarda progreso en localStorage tras cada voto y al mover navegacion cuando hay comentario en curso.
6. El CSV se puede exportar tanto durante la evaluacion (progreso) como en la pagina final.

## Interfaz y temas

- Incluye modo claro y modo oscuro con boton de cambio de tema en cabecera.
- La preferencia de tema queda guardada en localStorage.
- La galeria muestra miniaturas y estado por caso: `Match`, `No Match`, `Pendiente` y marca de comentario.

## Persistencia local

Se guardan en localStorage:

- resultados acumulados
- indice del caso actual

Si se cierra y reabre la pagina, la evaluacion se reanuda donde se dejo.

## Exportacion CSV

Cabecera exacta:

```text
id_imagen,tipo_polipo,id_modelo,veredicto,comentario_medico
```

Nombre de descarga:

- evaluacion_clinica_resultados.csv

## Uso rapido

1. Coloca tus imagenes en data/images/.
2. Genera o edita data/cases.json con el esquema correcto.
3. Abre index.html en el navegador.
4. Exporta el CSV cuando quieras (durante o al final de la evaluacion).

## Nota sobre file:// y CORS

Algunos navegadores bloquean `fetch("data/cases.json")` cuando se abre `index.html` con doble clic (`file://...`).

La app incluye un fallback:

1. En `file://`, la app muestra directamente una seccion "Carga Local Requerida".
2. Selecciona manualmente `clinical_eval_app/data/cases.json`.
3. La app cachea los casos en localStorage para futuras aperturas.

Alternativa recomendada para evitar esta limitacion: abrir la app con un servidor local (`http://localhost`).
