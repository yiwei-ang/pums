def generate_sql(person_table: str, housing_table: str) -> str:

    return f"""
WITH person_agg AS (
  SELECT
    SERIALNO,
    SUM(PWGTP) AS total_person_weight,
    SUM(PWGTP * AGEP) / NULLIF(SUM(PWGTP), 0) AS weighted_avg_age,
    SUM(CASE WHEN SEX = 2 THEN PWGTP ELSE 0 END) / NULLIF(SUM(PWGTP), 0) AS prop_female,
    SUM(CASE WHEN ESR IN (1, 2) THEN PWGTP ELSE 0 END) AS estimated_num_workers,
    SUM(PWGTP * INTP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_INTP,
    SUM(PWGTP * OIP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_OIP,
    SUM(PWGTP * PAP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_PAP,
    SUM(PWGTP * PERNP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_PERNP,
    SUM(PWGTP * PINCP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_PINCP,
    SUM(PWGTP * RETP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_RETP,
    SUM(PWGTP * SEMP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_SEMP,
    SUM(PWGTP * SSIP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_SSIP,
    SUM(PWGTP * SSP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_SSP,
    SUM(PWGTP * WAGP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) AS adj_avg_WAGP,
    SUM(PWGTP * PINCP * ADJINC / 1000000.0) / NULLIF(SUM(PWGTP), 0) / NULLIF(COUNT(*), 0) AS income_per_person,
    SUM(CASE WHEN DIS = 1 THEN PWGTP ELSE 0 END) AS num_disabled,
    SUM(CASE WHEN HICOV = 1 THEN PWGTP ELSE 0 END) / NULLIF(SUM(PWGTP), 0) AS prop_insured,
    SUM(CASE WHEN HINS4 = 1 THEN PWGTP ELSE 0 END) / NULLIF(SUM(PWGTP), 0) AS prop_medicaid,
    SUM(PWGTP * POVPIP) / NULLIF(SUM(PWGTP), 0) AS avg_poverty_ratio,
    SUM(CASE WHEN RACSOR = 1 THEN PWGTP ELSE 0 END) / NULLIF(SUM(PWGTP), 0) AS prop_other_race
  FROM {person_table}
  GROUP BY SERIALNO
)

SELECT
  h.*,
  h.ELEP * h.ADJHSG / 1000000.0 AS adj_ELEP,
  h.FULP * h.ADJHSG / 1000000.0 AS adj_FULP,
  h.GASP * h.ADJHSG / 1000000.0 AS adj_GASP,
  h.INSP * h.ADJHSG / 1000000.0 AS adj_INSP,
  h.MHP * h.ADJHSG / 1000000.0 AS adj_MHP,
  h.MRGP * h.ADJHSG / 1000000.0 AS adj_MRGP,
  h.RNTP * h.ADJHSG / 1000000.0 AS adj_RNTP,
  h.SMP * h.ADJHSG / 1000000.0 AS adj_SMP,
  h.SMOCP * h.ADJHSG / 1000000.0 AS adj_SMOCP,
  h.TAXAMT * h.ADJHSG / 1000000.0 AS adj_TAXAMT,
  h.VALP * h.ADJHSG / 1000000.0 AS adj_VALP,
  h.WATP * h.ADJHSG / 1000000.0 AS adj_WATP,
  h.HINCP / NULLIF(h.NP, 0) AS income_per_person_household,
  h.HINCP / NULLIF(h.RMSP, 0) AS income_per_room,
  h.RNTP / NULLIF(h.HINCP, 0) AS rent_to_income_ratio,
  h.HHLDRAGEP * h.HINCP AS age_x_income,
  p.*
FROM {housing_table} h
LEFT JOIN person_agg p ON h.SERIALNO = p.SERIALNO
WHERE h.FS IN (1, 2)
"""
