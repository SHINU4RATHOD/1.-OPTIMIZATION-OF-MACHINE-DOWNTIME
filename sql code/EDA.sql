SELECT 
--  mean
	AVG(Hydraulic_Pressure),  
	AVG(Coolant_Pressure),
	AVG(Air_System_Pressure),
	AVG(Coolant_Temperature),
	AVG(Hydraulic_Oil_Temperature),  
	AVG(Spindle_Bearing_Temperature), 
	AVG(Spindle_Vibration),  
	AVG(Tool_Vibration),  
	AVG(Spindle_Speed),  
	AVG(Voltage),  
	AVG(Torque), 
	AVG(Cutting),
    
-- median
	(
    SELECT AVG(Hydraulic_Pressure)
    FROM (
      SELECT Hydraulic_Pressure, 
             ROW_NUMBER() OVER (ORDER BY Hydraulic_Pressure) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Hydraulic_Pressure IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Hydraulic_Pressure,
  
  (
    SELECT AVG(Coolant_Pressure)
    FROM (
      SELECT Coolant_Pressure, 
             ROW_NUMBER() OVER (ORDER BY Coolant_Pressure) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Coolant_Pressure IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Coolant_Pressure,
  
  (
    SELECT AVG(Air_System_Pressure)
    FROM (
      SELECT Air_System_Pressure, 
             ROW_NUMBER() OVER (ORDER BY Air_System_Pressure) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Air_System_Pressure IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Air_System_Pressure,
  
  (
    SELECT AVG(Coolant_Temperature)
    FROM (
      SELECT Coolant_Temperature, 
             ROW_NUMBER() OVER (ORDER BY Coolant_Temperature) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Coolant_Temperature IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Coolant_Temperature,
  
  (
    SELECT AVG(Hydraulic_Oil_Temperature)
    FROM (
      SELECT Hydraulic_Oil_Temperature, 
             ROW_NUMBER() OVER (ORDER BY Hydraulic_Oil_Temperature) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Hydraulic_Oil_Temperature IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Hydraulic_Oil_Temperature,
  
  (
    SELECT AVG(Spindle_Bearing_Temperature)
    FROM (
      SELECT Spindle_Bearing_Temperature, 
             ROW_NUMBER() OVER (ORDER BY Spindle_Bearing_Temperature) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Spindle_Bearing_Temperature IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Spindle_Bearing_Temperature,
  
  (
    SELECT AVG(Spindle_Vibration)
    FROM (
      SELECT Spindle_Vibration, 
             ROW_NUMBER() OVER (ORDER BY Spindle_Vibration) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Spindle_Vibration IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Spindle_Vibration,
  
  (
    SELECT AVG(Tool_Vibration)
    FROM (
      SELECT Tool_Vibration, 
             ROW_NUMBER() OVER (ORDER BY Tool_Vibration) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Tool_Vibration IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Tool_Vibration,
  
  (
    SELECT AVG(Spindle_Speed)
    FROM (
      SELECT Spindle_Speed, 
             ROW_NUMBER() OVER (ORDER BY Spindle_Speed) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Spindle_Speed IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Spindle_Speed,
  
  (
    SELECT AVG(Voltage)
    FROM (
      SELECT Voltage, 
             ROW_NUMBER() OVER (ORDER BY Voltage) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Voltage IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Voltage,
  
  (
    SELECT AVG(Torque)
    FROM (
      SELECT Torque, 
             ROW_NUMBER() OVER (ORDER BY Torque) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Torque IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Torque,
  
  (
    SELECT AVG(Cutting)
    FROM (
      SELECT Cutting, 
             ROW_NUMBER() OVER (ORDER BY Cutting) AS rn,
             COUNT(*) OVER () AS total_count
      FROM dummymd
      WHERE Cutting IS NOT NULL
    ) AS subquery
    WHERE rn = (total_count + 1) / 2 OR rn = (total_count + 2) / 2
  ) AS median_Cutting,
  
  -- 3. Mode
    (
        SELECT Assembly_Line_No
        FROM dummymd
        GROUP BY Assembly_Line_No
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS mode_Category1,
    
    (
        SELECT Downtime
        FROM dummymd
        GROUP BY Downtime
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS mode_Category2,
    
    -- Variance
    VARIANCE(Hydraulic_Pressure) AS variance_Hydraulic_Pressure,
    VARIANCE(Coolant_Pressure) AS variance_Coolant_Pressure,
    VARIANCE(Air_System_Pressure) AS variance_Air_System_Pressure,
    VARIANCE(Coolant_Temperature) AS variance_Coolant_Temperature,
    VARIANCE(Hydraulic_Oil_Temperature) AS variance_Hydraulic_Oil_Temperature,
    VARIANCE(Spindle_Bearing_Temperature) AS variance_Spindle_Bearing_Temperature,
    VARIANCE(Spindle_Vibration) AS variance_Spindle_Vibration,
    VARIANCE(Tool_Vibration) AS variance_Tool_Vibration,
    VARIANCE(Spindle_Speed) AS variance_Spindle_Speed,
    VARIANCE(Voltage) AS variance_Voltage,
    VARIANCE(Torque) AS variance_Torque,
    VARIANCE(Cutting) AS variance_Cutting,
    
    -- Standard deviation
    STDDEV(Hydraulic_Pressure) AS std_dev_Hydraulic_Pressure,
    STDDEV(Coolant_Pressure) AS std_dev_Coolant_Pressure,
    STDDEV(Air_System_Pressure) AS std_dev_Air_System_Pressure,
    STDDEV(Coolant_Temperature) AS std_dev_Coolant_Temperature,
    STDDEV(Hydraulic_Oil_Temperature) AS std_dev_Hydraulic_Oil_Temperature,
    STDDEV(Spindle_Bearing_Temperature) AS std_dev_Spindle_Bearing_Temperature,
    STDDEV(Spindle_Vibration) AS std_dev_Spindle_Vibration,
    STDDEV(Tool_Vibration) AS std_dev_Tool_Vibration,
    STDDEV(Spindle_Speed) AS std_dev_Spindle_Speed,
    STDDEV(Voltage) AS std_dev_Voltage,
    STDDEV(Torque) AS std_dev_Torque,
    STDDEV(Cutting) AS std_dev_Cutting,
    
    -- Range
    MAX(Hydraulic_Pressure) - MIN(Hydraulic_Pressure) AS range_Hydraulic_Pressure,
    MAX(Coolant_Pressure) - MIN(Coolant_Pressure) AS range_Coolant_Pressure,
    MAX(Air_System_Pressure) - MIN(Air_System_Pressure) AS range_Air_System_Pressure,
    MAX(Coolant_Temperature) - MIN(Coolant_Temperature) AS range_Coolant_Temperature,
    MAX(Hydraulic_Oil_Temperature) - MIN(Hydraulic_Oil_Temperature) AS range_Hydraulic_Oil_Temperature,
    MAX(Spindle_Bearing_Temperature) - MIN(Spindle_Bearing_Temperature) AS range_Spindle_Bearing_Temperature,
    MAX(Spindle_Vibration) - MIN(Spindle_Vibration) AS range_Spindle_Vibration,
    MAX(Tool_Vibration) - MIN(Tool_Vibration) AS range_Tool_Vibration,
    MAX(Spindle_Speed) - MIN(Spindle_Speed) AS range_Spindle_Speed,
    MAX(Voltage) - MIN(Voltage) AS range_Voltage,
    MAX(Torque) - MIN(Torque) AS range_Torque,
    MAX(Cutting) - MIN(Cutting) AS range_Cutting,
    
    -- skewness and kurtosis
    (SUM(POW(Hydraulic_Pressure - mean_Hydraulic_Pressure, 3)) / (COUNT(Hydraulic_Pressure) * POW(STDDEV(Hydraulic_Pressure), 3))) AS Skewness_Hydraulic_Pressure,
    (SUM(POW(Hydraulic_Pressure - mean_Hydraulic_Pressure, 4)) / (COUNT(Hydraulic_Pressure) * POW(STDDEV(Hydraulic_Pressure), 4))) AS Kurtosis_Hydraulic_Pressure,
    
    (SUM(POW(Coolant_Pressure - mean_Coolant_Pressure, 3)) / (COUNT(Coolant_Pressure) * POW(STDDEV(Coolant_Pressure), 3))) AS Skewness_Coolant_Pressure,
    (SUM(POW(Coolant_Pressure - mean_Coolant_Pressure, 4)) / (COUNT(Coolant_Pressure) * POW(STDDEV(Coolant_Pressure), 4))) AS Kurtosis_Coolant_Pressure,
    
    (SUM(POW(Air_System_Pressure - mean_Air_System_Pressure, 3)) / (COUNT(Air_System_Pressure) * POW(STDDEV(Air_System_Pressure), 3))) AS Skewness_Air_System_Pressure,
    (SUM(POW(Air_System_Pressure - mean_Air_System_Pressure, 4)) / (COUNT(Air_System_Pressure) * POW(STDDEV(Air_System_Pressure), 4))) AS Kurtosis_Air_System_Pressure,
    
    (SUM(POW(Coolant_Temperature - mean_Coolant_Temperature, 3)) / (COUNT(Coolant_Temperature) * POW(STDDEV(Coolant_Temperature), 3))) AS Skewness_Coolant_Temperature,
    (SUM(POW(Coolant_Temperature - mean_Coolant_Temperature, 4)) / (COUNT(Coolant_Temperature) * POW(STDDEV(Coolant_Temperature), 4))) AS Kurtosis_Coolant_Temperature,
    
    (SUM(POW(Hydraulic_Oil_Temperature - mean_Hydraulic_Oil_Temperature, 3)) / (COUNT(Hydraulic_Oil_Temperature) * POW(STDDEV(Hydraulic_Oil_Temperature), 3))) AS Skewness_Hydraulic_Oil_Temperature,
    (SUM(POW(Hydraulic_Oil_Temperature - mean_Hydraulic_Oil_Temperature, 4)) / (COUNT(Hydraulic_Oil_Temperature) * POW(STDDEV(Hydraulic_Oil_Temperature), 4))) AS Kurtosis_Hydraulic_Oil_Temperature,
    
    (SUM(POW(Spindle_Bearing_Temperature - mean_Spindle_Bearing_Temperature, 3)) / (COUNT(Spindle_Bearing_Temperature) * POW(STDDEV(Spindle_Bearing_Temperature), 3))) AS Skewness_Spindle_Bearing_Temperature,
    (SUM(POW(Spindle_Bearing_Temperature - mean_Spindle_Bearing_Temperature, 4)) / (COUNT(Spindle_Bearing_Temperature) * POW(STDDEV(Spindle_Bearing_Temperature), 4))) AS Kurtosis_Spindle_Bearing_Temperature,
    
    (SUM(POW(Spindle_Vibration - mean_Spindle_Vibration, 3)) / (COUNT(Spindle_Vibration) * POW(STDDEV(Spindle_Vibration), 3))) AS Skewness_Spindle_Vibration,
    (SUM(POW(Spindle_Vibration - mean_Spindle_Vibration, 4)) / (COUNT(Spindle_Vibration) * POW(STDDEV(Spindle_Vibration), 4))) AS Kurtosis_Spindle_Vibration,
    
    (SUM(POW(Tool_Vibration - mean_Tool_Vibration, 3)) / (COUNT(Tool_Vibration) * POW(STDDEV(Tool_Vibration), 3))) AS Skewness_Tool_Vibration,
    (SUM(POW(Tool_Vibration - mean_Tool_Vibration, 4)) / (COUNT(Tool_Vibration) * POW(STDDEV(Tool_Vibration), 4))) AS Kurtosis_Tool_Vibration,
    
    (SUM(POW(Spindle_Speed - mean_Spindle_Speed, 3)) / (COUNT(Spindle_Speed) * POW(STDDEV(Spindle_Speed), 3))) AS Skewness_Spindle_Speed,
    (SUM(POW(Spindle_Speed - mean_Spindle_Speed, 4)) / (COUNT(Spindle_Speed) * POW(STDDEV(Spindle_Speed), 4))) AS Kurtosis_Spindle_Speed,
    
    (SUM(POW(Voltage - mean_Voltage, 3)) / (COUNT(Voltage) * POW(STDDEV(Voltage), 3))) AS Skewness_Voltage,
    (SUM(POW(Voltage - mean_Voltage, 4)) / (COUNT(Voltage) * POW(STDDEV(Voltage), 4))) AS Kurtosis_Voltage,
    
    (SUM(POW(Torque - mean_Torque, 3)) / (COUNT(Torque) * POW(STDDEV(Torque), 3))) AS Skewness_Torque,
    (SUM(POW(Torque - mean_Torque, 4)) / (COUNT(Torque) * POW(STDDEV(Torque), 4))) AS Kurtosis_Torque,
    
    (SUM(POW(Cutting - mean_Cutting, 3)) / (COUNT(Cutting) * POW(STDDEV(Cutting), 3))) AS Skewness_Cutting,
    (SUM(POW(Cutting - mean_Cutting, 4)) / (COUNT(Cutting) * POW(STDDEV(Cutting), 4))) AS Kurtosis_Cutting
    
FROM (
    SELECT 
        AVG(Hydraulic_Pressure) AS mean_Hydraulic_Pressure,
        STDDEV(Hydraulic_Pressure) AS stddev_Hydraulic_Pressure,
        
        AVG(Coolant_Pressure) AS mean_Coolant_Pressure,
        STDDEV(Coolant_Pressure) AS stddev_Coolant_Pressure,
        
        AVG(Air_System_Pressure) AS mean_Air_System_Pressure,
        STDDEV(Air_System_Pressure) AS stddev_Air_System_Pressure,
        
        AVG(Coolant_Temperature) AS mean_Coolant_Temperature,
        STDDEV(Coolant_Temperature) AS stddev_Coolant_Temperature,
        
        AVG(Hydraulic_Oil_Temperature) AS mean_Hydraulic_Oil_Temperature,
        STDDEV(Hydraulic_Oil_Temperature) AS stddev_Hydraulic_Oil_Temperature,
        
        AVG(Spindle_Bearing_Temperature) AS mean_Spindle_Bearing_Temperature,
        STDDEV(Spindle_Bearing_Temperature) AS stddev_Spindle_Bearing_Temperature,
        
        AVG(Spindle_Vibration) AS mean_Spindle_Vibration,
        STDDEV(Spindle_Vibration) AS stddev_Spindle_Vibration,
        
        AVG(Tool_Vibration) AS mean_Tool_Vibration,
        STDDEV(Tool_Vibration) AS stddev_Tool_Vibration,
        
        AVG(Spindle_Speed) AS mean_Spindle_Speed,
        STDDEV(Spindle_Speed) AS stddev_Spindle_Speed,
        
        AVG(Voltage) AS mean_Voltage,
        STDDEV(Voltage) AS stddev_Voltage,
        
        AVG(Torque) AS mean_Torque,
        STDDEV(Torque) AS stddev_Torque,
        
        AVG(Cutting) AS mean_Cutting,
        STDDEV(Cutting) AS stddev_Cutting
        
    FROM dummymd
) AS subquery, dummymd;