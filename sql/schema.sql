-- ============================================================
-- schema.sql
-- Traffic Accident Severity Analytics — Database Schema
-- Compatible with: SQLite 3.x and PostgreSQL 14+
-- ============================================================

-- Drop tables in reverse dependency order (safe for re-runs)
DROP TABLE IF EXISTS accidents;
DROP TABLE IF EXISTS weather_grid;
DROP TABLE IF EXISTS road_context_lookup;

-- ── road_context_lookup ───────────────────────────────────────────────────
-- Stores the DfT STATS19 code-to-label mapping (third data source).
CREATE TABLE road_context_lookup (
    id          INTEGER     PRIMARY KEY AUTOINCREMENT,  -- use SERIAL for PostgreSQL
    code_type   TEXT        NOT NULL,   -- e.g. road_type, speed_limit
    code_value  INTEGER     NOT NULL,
    label       TEXT        NOT NULL,
    UNIQUE(code_type, code_value)
);

-- ── weather_grid ──────────────────────────────────────────────────────────
-- One row per grid-cell × date (Open-Meteo source 2).
CREATE TABLE weather_grid (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    grid_latitude   REAL        NOT NULL,
    grid_longitude  REAL        NOT NULL,
    obs_date        TEXT        NOT NULL,   -- ISO 8601 YYYY-MM-DD
    temp_mean_c     REAL,
    precip_mm       REAL,
    wind_max_kmh    REAL
);

CREATE UNIQUE INDEX uix_weather_grid ON weather_grid
    (grid_latitude, grid_longitude, obs_date);

-- ── accidents ─────────────────────────────────────────────────────────────
-- One row per collision (DfT source 1 + joined weather + road context).
CREATE TABLE accidents (
    collision_index         TEXT        PRIMARY KEY,
    accident_date           TEXT,       -- ISO 8601 YYYY-MM-DD
    day_of_week             INTEGER,    -- 0=Monday … 6=Sunday
    month                   INTEGER,
    year                    INTEGER,
    hour_of_day             REAL,
    is_weekend              INTEGER,    -- 0 or 1
    latitude                REAL,
    longitude               REAL,
    collision_severity      INTEGER,    -- 1=Fatal 2=Serious 3=Slight (verify in data dictionary)
    number_of_vehicles      INTEGER,
    number_of_casualties    INTEGER,
    road_type               INTEGER,
    road_type_label         TEXT,
    speed_limit             INTEGER,
    speed_limit_label       TEXT,
    light_conditions        INTEGER,
    light_conditions_label  TEXT,
    weather_conditions      INTEGER,
    weather_conditions_label TEXT,
    road_surface_conditions INTEGER,
    road_surface_conditions_label TEXT,
    junction_detail         INTEGER,
    junction_detail_label   TEXT,
    urban_or_rural_area     INTEGER,
    urban_or_rural_area_label TEXT,
    -- Joined weather fields
    grid_latitude           REAL,
    grid_longitude          REAL,
    wx_temp_mean_c          REAL,
    wx_precip_mm            REAL,
    wx_wind_max_kmh         REAL
);

-- ── Example analytics queries ─────────────────────────────────────────────

-- Severity distribution
-- SELECT collision_severity, COUNT(*) AS n
-- FROM accidents
-- GROUP BY collision_severity
-- ORDER BY collision_severity;

-- Monthly accident counts
-- SELECT month, COUNT(*) AS n
-- FROM accidents
-- GROUP BY month
-- ORDER BY month;

-- Average weather by severity
-- SELECT collision_severity,
--        ROUND(AVG(wx_temp_mean_c), 1)  AS avg_temp_c,
--        ROUND(AVG(wx_precip_mm), 2)    AS avg_precip_mm,
--        ROUND(AVG(wx_wind_max_kmh), 1) AS avg_wind_kmh
-- FROM accidents
-- GROUP BY collision_severity;

-- Severity by road type
-- SELECT road_type_label, collision_severity, COUNT(*) AS n
-- FROM accidents
-- GROUP BY road_type_label, collision_severity
-- ORDER BY road_type_label, collision_severity;

-- Peak accident hours
-- SELECT CAST(hour_of_day AS INTEGER) AS hour, COUNT(*) AS n
-- FROM accidents
-- WHERE hour_of_day IS NOT NULL
-- GROUP BY hour
-- ORDER BY hour;
