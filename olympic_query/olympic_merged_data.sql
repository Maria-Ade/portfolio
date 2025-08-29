SELECT o.Season,
       o.Year,
       o.City,
       o.Sport,
       o.Discipline,
       o.Athlete,
       d.country AS Country,
       o.Country AS Country_code,
       d.Population,
       d.GDP_per_Capita,
       o.Gender,
       o.Event,
       o.Medal
FROM (
    SELECT 'Winter' AS Season, Year, City, Sport, Discipline, Athlete, Country, Gender, Event, Medal
    FROM [Project].[dbo].[sport_winter]
    UNION ALL
    SELECT 'Summer' AS Season, Year, City, Sport, Discipline, Athlete, Country, Gender, Event, Medal
    FROM [Project].[dbo].[sport_summer]
) AS o
LEFT JOIN [Project].[dbo].[sport_dictionary] AS d
       ON o.Country = d.Code;
