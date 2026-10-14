-- ============================================================
-- analytics_queries.sql
-- Useful SQL queries for the Traffic Accident dataset
-- Compatible with SQLite 3.x (also valid PostgreSQL with minor adjustments)
-- Run with: sqlite3 data/traffic_accidents.db < sql/analytics_queries.sql
-- ============================================================

-- ── 1. Overall severity distribution ─────────────────────────────────────
SELECT
    collision_severity,
    CASE collision_severity
        WHEN 1 THEN 'Fatal'
        WHEN 2 THEN 'Serious'
        WHEN 3 THEN 'Slight'
        ELSE 'Unknown'
    END AS severity_name,
    COUNT(*) AS accident_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM accidents
GROUP BY collision_severity
ORDER BY collision_severity;

-- ── 2. Accidents by month ─────────────────────────────────────────────────
SELECT
    month,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    SUM(CASE WHEN collision_severity = 2 THEN 1 ELSE 0 END) AS serious,
    SUM(CASE WHEN collision_severity = 3 THEN 1 ELSE 0 END) AS slight
FROM accidents
WHERE month IS NOT NULL
GROUP BY month
ORDER BY month;

-- ── 3. Accidents by hour of day (peak hours) ──────────────────────────────
SELECT
    CAST(hour_of_day AS INTEGER) AS hour,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal
FROM accidents
WHERE hour_of_day IS NOT NULL
GROUP BY CAST(hour_of_day AS INTEGER)
ORDER BY hour;

-- ── 4. Road type vs severity ──────────────────────────────────────────────
SELECT
    COALESCE(road_type_label, 'Unknown') AS road_type,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY road_type_label
ORDER BY fatal_pct DESC;

-- ── 5. Average weather by severity ───────────────────────────────────────
SELECT
    collision_severity,
    CASE collision_severity WHEN 1 THEN 'Fatal' WHEN 2 THEN 'Serious' WHEN 3 THEN 'Slight' END AS severity_name,
    COUNT(*) AS n,
    ROUND(AVG(wx_temp_mean_c), 1)   AS avg_temp_c,
    ROUND(AVG(wx_precip_mm), 2)     AS avg_precip_mm,
    ROUND(AVG(wx_wind_max_kmh), 1)  AS avg_wind_kmh
FROM accidents
WHERE wx_temp_mean_c IS NOT NULL
GROUP BY collision_severity
ORDER BY collision_severity;

-- ── 6. Weather conditions vs severity ────────────────────────────────────
SELECT
    COALESCE(weather_conditions_label, 'Unknown') AS weather_condition,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY weather_conditions_label
ORDER BY total_accidents DESC;

-- ── 7. Speed limit breakdown ──────────────────────────────────────────────
SELECT
    COALESCE(speed_limit_label, CAST(speed_limit AS TEXT), 'Unknown') AS speed_limit,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY speed_limit_label, speed_limit
ORDER BY fatal_pct DESC;

-- ── 8. Urban vs rural ────────────────────────────────────────────────────
SELECT
    COALESCE(urban_or_rural_area_label, 'Unknown') AS area_type,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY urban_or_rural_area_label
ORDER BY total_accidents DESC;

-- ── 9. Weekend vs weekday ────────────────────────────────────────────────
SELECT
    CASE is_weekend WHEN 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY is_weekend;

-- ── 10. Top 10 dangerous junctions ───────────────────────────────────────
SELECT
    COALESCE(junction_detail_label, 'Unknown') AS junction_type,
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY junction_detail_label
ORDER BY total_accidents DESC
LIMIT 10;

-- ── 11. Weather match rate ────────────────────────────────────────────────
SELECT
    COUNT(*) AS total_accidents,
    SUM(CASE WHEN wx_temp_mean_c IS NOT NULL THEN 1 ELSE 0 END) AS weather_matched,
    ROUND(100.0 * SUM(CASE WHEN wx_temp_mean_c IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS match_pct
FROM accidents;

-- ── 12. Surface conditions vs severity ───────────────────────────────────
SELECT
    COALESCE(road_surface_conditions_label, 'Unknown') AS surface,
    COUNT(*) AS total,
    SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) AS fatal,
    ROUND(100.0 * SUM(CASE WHEN collision_severity = 1 THEN 1 ELSE 0 END) / COUNT(*), 3) AS fatal_pct
FROM accidents
GROUP BY road_surface_conditions_label
ORDER BY fatal_pct DESC;
