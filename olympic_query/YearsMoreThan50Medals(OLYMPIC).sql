SELECT TOP 16 Country,
       COUNT(*) AS YearsMoreThan50Medals 
FROM (
    SELECT [Year], Country, COUNT(*) AS medal_count
    FROM [Project].[dbo].[olympic_summary]
    WHERE Country IS NOT NULL AND Medal IS NOT NULL
    GROUP BY [Year], Country
) cm
WHERE cm.medal_count > 50
GROUP BY Country
ORDER BY YearsMoreThan50Medals DESC;
