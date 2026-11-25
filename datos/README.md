# datos

La serie del grupo es la demanda mensual de energía eléctrica de Colombia, la
misma del miniproyecto 1, con la misma ficha.

| Campo | Valor |
|---|---|
| Archivo | `demanda_mensual.csv` |
| Qué mide | Demanda real de energía eléctrica del Sistema Interconectado Nacional (SIN) de Colombia, en GWh por mes. Es la suma de la demanda horaria de todo el sistema (lo que consumen los usuarios regulados y no regulados más las pérdidas de la red), agregada por mes. |
| Fuente | XM S.A. E.S.P., operador del SIN y administrador del mercado de energía mayorista de Colombia. Se descargó la demanda real horaria del sistema por medio de la API pública de XM, con el cliente oficial `pydataxm` (<https://github.com/EquipoAnaliticaXM/API_XM>), y se agregó a meses. Los mismos datos se consultan en el portal SINERGOX (<https://sinergox.xm.com.co>). |
| Licencia | La API de XM es de acceso público, sin registro ni clave. El cliente oficial `pydataxm` está bajo licencia MIT (© 2020 XM S.A. E.S.P., <https://github.com/EquipoAnaliticaXM/API_XM/blob/master/LICENSE>). Los datos se usan con fines académicos, sin redistribución comercial, citando a XM como fuente. |
| Frecuencia | mensual |
| Periodo | 2015-01 a 2025-12 (132 observaciones, 11 años completos) |
| Columna de fecha | `ts` (primer día del mes, `AAAA-MM-DD`) |
| Columna que se pronostica | `GWh` |

La tercera columna, `horas_con_dato`, es el número de horas del mes que
tenían dato en la fuente (744, 720 o 672 según el mes). Sirve para comprobar
que ningún mes está incompleto: en los 132 meses coincide con las horas del
mes, así que la serie no tiene huecos ni meses parciales.

## Por qué cumple las condiciones

- **Mensual**, 132 meses: más del doble de las 60 observaciones mínimas.
  El *backtest* usa los últimos 24 meses (2024-01 a 2025-12) y en su primer
  origen quedan 108 meses para ajustar.
- **Pública**, con fuente citable (XM) y acceso abierto.
- **Estacionalidad visible**: febrero es el mes más bajo (tiene 28 días) y
  agosto el más alto; la estacional de la STL oscila entre $-439$ y $+186$
  GWh.
- **Tendencia creciente**: de unos 5 500 GWh mensuales en 2015 a unos
  7 000 en 2025, con la caída de la cuarentena de 2020 como atípico.
