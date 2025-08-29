WITH per_year AS (
  SELECT
    Country = COALESCE(NULLIF(LTRIM(RTRIM(Country)), ''), 'Unknown'),
    [Year],
    MedalCount = COUNT(*),
    Pop = MAX(Population)        -- population for that country in that year
  FROM [Project].[dbo].[olympic_summary]
  WHERE Medal IS NOT NULL
  GROUP BY COALESCE(NULLIF(LTRIM(RTRIM(Country)), ''), 'Unknown'), [Year]
)
SELECT TOP (15)
  Country,
  TotalMedals = SUM(MedalCount),
  AvgMedalsPerMillion = AVG(CASE WHEN Pop>0 THEN MedalCount * 1e6 / Pop END)
FROM per_year
WHERE Pop IS NOT NULL
GROUP BY Country
HAVING SUM(MedalCount) >= 10
ORDER BY AvgMedalsPerMillion DESC, TotalMedals DESC;
