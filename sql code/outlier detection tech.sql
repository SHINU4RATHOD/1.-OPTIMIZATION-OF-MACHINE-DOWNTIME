# 1. Z-Scores:
SELECT *
FROM machine_downtime
WHERE ABS( (`Hydraulic_Pressure(bar)` - (SELECT AVG(`Hydraulic_Pressure(bar)`) FROM machine_downtime)) / (SELECT STDDEV(`Hydraulic_Pressure(bar)`) FROM machine_downtime)) > 3;


# 2. Interquartile Range (IQR):
SELECT *
FROM machine_downtime
WHERE `Hydraulic_Pressure(bar)` < (
    SELECT PERCENTILE_CONT(0.25)
    FROM (
        SELECT `Hydraulic_Pressure(bar)`
        FROM machine_downtime
        ORDER BY `Hydraulic_Pressure(bar)`
    ) AS ordered_data
) - 1.5 * (
    SELECT PERCENTILE_IQR(0.75)
    FROM (
        SELECT `Hydraulic_Pressure(bar)`
        FROM machine_downtime
        ORDER BY `Hydraulic_Pressure(bar)`
    ) AS ordered_data
)
OR `Hydraulic_Pressure(bar)` > (
    SELECT PERCENTILE_CONT(0.75)
    FROM (
        SELECT `Hydraulic_Pressure(bar)`
        FROM machine_downtime
        ORDER BY `Hydraulic_Pressure(bar)`
    ) AS ordered_data
) + 1.5 * (
    SELECT PERCENTILE_IQR(0.75)
    FROM (
        SELECT `Hydraulic_Pressure(bar)`
        FROM machine_downtime
        ORDER BY `Hydraulic_Pressure(bar)`
    ) AS ordered_data
);





# 3.  Using Window Functions:NTILE():
SELECT *
FROM (
    SELECT *, NTILE(4) OVER (ORDER BY `Hydraulic_Pressure(bar)`) AS quartile
    FROM machine_downtime
) AS ranked_data
WHERE quartile = 1 OR quartile = 4; -- Suspected outliers in extreme quartiles



# examining the data distribution
SELECT `Hydraulic_Pressure(bar)`, COUNT(*) AS frequency
FROM machine_downtime
GROUP BY `Hydraulic_Pressure(bar)`
ORDER BY `Hydraulic_Pressure(bar)`;



# 4. Examining Data Distribution: histogram
SELECT value, COUNT(*) AS frequency
FROM your_table
GROUP BY value
ORDER BY value;



