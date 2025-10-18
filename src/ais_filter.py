from pathlib import Path
import warnings
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow as pa
from functools import reduce



warnings.filterwarnings('ignore')

def create_filtered_ais(ais_path, dst_path, lon_min, lon_max, lat_min, lat_max, sog_min, sog_max):
    pf = pq.ParquetFile(ais_path)
    print("Row groups :", pf.num_row_groups)

    # Delete old file
    Path(dst_path).unlink(missing_ok=True)

    writer = None

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)

        # --- If LON € [-180, 180] Else apply---
        # if tbl.schema.field('LON').type == pa.float64():
        #     # convert 0…360 -> -180…180
        #     lon = pc.add(tbl['LON'], pa.scalar(180.0))
        #     lon = pc.remainder(lon, pa.scalar(360.0))
        #     lon = pc.subtract(lon, pa.scalar(180.0))
        #     tbl = tbl.set_column(tbl.schema.get_field_index('LON'), 'LON', lon)

        # --- Apply filters ---

        mmsi_ok = pc.and_(
            pc.is_valid(tbl["MMSI"]),
            pc.and_(
                pc.not_equal(tbl["MMSI"], pa.scalar(0)),
                pc.and_(
                    pc.greater_equal(tbl["MMSI"], pa.scalar(100_000_000)),
                    pc.less_equal(tbl["MMSI"], pa.scalar(999_999_999))
                )
            )
        )

        lon_ok = pc.and_(
            pc.greater_equal(tbl['LON'], pa.scalar(lon_min)),
            pc.less_equal(tbl['LON'],  pa.scalar(lon_max))
        )

        lat_ok = pc.and_(
            pc.greater_equal(tbl['LAT'], pa.scalar(lat_min)),
            pc.less_equal(tbl['LAT'],  pa.scalar(lat_max))
        )

        sog_ok = pc.and_(
            pc.greater_equal(tbl['SOG'], pa.scalar(sog_min)),
            pc.less_equal(tbl['SOG'],  pa.scalar(sog_max))
        )

        heading_ok = pc.and_(
            pc.greater_equal(tbl['Heading'], pa.scalar(0)),
            pc.less(tbl['Heading'],  pa.scalar(360))
        )


        cog_ok = pc.and_(
                pc.greater_equal(tbl['COG'], pa.scalar(0)),
                pc.less(tbl['COG'],  pa.scalar(360))
            )

        status_ok = pc.equal(tbl['Status'], pa.scalar(0))

        conditions = [lon_ok, lat_ok, sog_ok, heading_ok, cog_ok, status_ok]
        mask = reduce(pc.and_kleene, conditions)

        tbl_f = tbl.filter(mask)

        if tbl_f.num_rows == 0:
            print(f"RG {rg}: 0 filtered line — ignore")
            continue

        subset = None
        names = subset or tbl_f.schema.names

        mask_nan = None
        for name in names:
            col = tbl_f[name]

            # Keep VALID value
            keep = pc.is_valid(col)

            # if float: remove NaN and ±inf
            if pa.types.is_floating(col.type):
                not_nan = pc.invert(pc.is_nan(col))
                finite  = pc.is_finite(col)
                keep = pc.and_kleene(keep, pc.and_kleene(not_nan, finite))

            mask_nan = keep if mask_nan is None else pc.and_kleene(mask_nan, keep)

        tbl_clean = tbl_f.filter(mask_nan)

        if writer is None:
            writer = pq.ParquetWriter(dst_path, schema=tbl_f.schema, compression="snappy")
        writer.write_table(tbl_clean)
        print(f"RG {rg}: write {tbl_f.num_rows} filtered lines")

    if writer is not None:
        writer.close()
        print(f"Create file : {dst_path}")
    else:
        print("Data are not in the filter — No created file.")
