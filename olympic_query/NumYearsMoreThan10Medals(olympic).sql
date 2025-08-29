SELECT Country,
       COUNT(*) AS NumYearsMoreThan10Medals
FROM (
    SELECT [Year], Country, COUNT(*) AS medal_count
    FROM [Project].[dbo].[olympic_summary]
    WHERE Medal IS NOT NULL
    GROUP BY [Year], Country
) cm
WHERE cm.medal_count > 10
GROUP BY Country
ORDER BY NumYearsMoreThan10Medals DESC