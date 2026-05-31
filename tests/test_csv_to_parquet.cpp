// Round-trip checks for the one-off CSV/TSV -> Parquet converter.
// The converter must preserve column types (integer / floating / string),
// column names (header passthrough or synthesized positional names), row
// order, and the user-supplied null sentinel.

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>

#include "a3i/tools/csv_to_parquet.hpp"

namespace fs = std::filesystem;

namespace {

class TempDir {
   public:
    TempDir() {
        path_ = fs::temp_directory_path() /
                ("a3i_pq_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                 "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path operator/(std::string_view leaf) const { return path_ / leaf; }

   private:
    fs::path path_;
};

void write_text(const fs::path& p, std::string_view content) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::shared_ptr<arrow::Table> read_parquet(const fs::path& p) {
    auto infile = arrow::io::ReadableFile::Open(p.string()).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
    return reader->ReadTable().ValueOrDie();
}

}  // namespace

// A headered CSV keeps its column names and infers int / double / string.
TEST(CsvToParquet, HeaderedTypesNamesAndOrder) {
    TempDir tmp;
    const auto csv = tmp / "in.csv";
    write_text(csv,
        "id,price,label\n"
        "1,2.5,foo\n"
        "2,,bar\n"        // empty price -> null
        "3,4.0,NA\n");    // null sentinel in label

    a3i::CsvToParquetOptions opts;
    opts.input_path  = csv;
    opts.output_path = tmp / "out.parquet";
    opts.has_header  = true;
    opts.null_string = "NA";
    opts.overwrite   = true;
    const auto rep = a3i::csv_to_parquet(opts);

    EXPECT_EQ(rep.rows, 3u);
    ASSERT_EQ(rep.column_names.size(), 3u);
    EXPECT_EQ(rep.column_names[0], "id");
    EXPECT_EQ(rep.column_names[1], "price");
    EXPECT_EQ(rep.column_names[2], "label");

    auto table = read_parquet(opts.output_path);
    ASSERT_EQ(table->num_columns(), 3);
    ASSERT_EQ(table->num_rows(), 3);

    const auto& schema = *table->schema();
    EXPECT_EQ(schema.field(0)->name(), "id");
    EXPECT_EQ(schema.field(1)->name(), "price");
    EXPECT_EQ(schema.field(2)->name(), "label");
    EXPECT_EQ(schema.field(0)->type()->id(), arrow::Type::INT64);
    EXPECT_EQ(schema.field(1)->type()->id(), arrow::Type::DOUBLE);
    EXPECT_EQ(schema.field(2)->type()->id(), arrow::Type::STRING);

    auto ids = std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
    EXPECT_EQ(ids->Value(0), 1);
    EXPECT_EQ(ids->Value(1), 2);
    EXPECT_EQ(ids->Value(2), 3);

    auto price = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    EXPECT_DOUBLE_EQ(price->Value(0), 2.5);
    EXPECT_TRUE(price->IsNull(1));
    EXPECT_DOUBLE_EQ(price->Value(2), 4.0);

    auto label = std::static_pointer_cast<arrow::StringArray>(table->column(2)->chunk(0));
    EXPECT_EQ(label->GetString(0), "foo");
    EXPECT_EQ(label->GetString(1), "bar");
    EXPECT_TRUE(label->IsNull(2));
}

// A headerless input gets positional names col0, col1, ...
TEST(CsvToParquet, HeaderlessSynthesizesPositionalNames) {
    TempDir tmp;
    const auto csv = tmp / "in.csv";
    write_text(csv,
        "10,1.5\n"
        "20,2.5\n");

    a3i::CsvToParquetOptions opts;
    opts.input_path  = csv;
    opts.output_path = tmp / "out.parquet";
    opts.has_header  = false;
    opts.overwrite   = true;
    const auto rep = a3i::csv_to_parquet(opts);

    ASSERT_EQ(rep.column_names.size(), 2u);
    EXPECT_EQ(rep.column_names[0], "col0");
    EXPECT_EQ(rep.column_names[1], "col1");

    auto table = read_parquet(opts.output_path);
    EXPECT_EQ(table->schema()->field(0)->name(), "col0");
    EXPECT_EQ(table->schema()->field(1)->name(), "col1");
    EXPECT_EQ(table->num_rows(), 2);
}

// A tab-separated input round-trips when the delimiter is supplied.
TEST(CsvToParquet, TabDelimited) {
    TempDir tmp;
    const auto csv = tmp / "in.tsv";
    write_text(csv,
        "a\tb\n"
        "1\t2\n"
        "3\t4\n");

    a3i::CsvToParquetOptions opts;
    opts.input_path  = csv;
    opts.output_path = tmp / "out.parquet";
    opts.has_header  = true;
    opts.delimiter   = '\t';
    opts.overwrite   = true;
    const auto rep = a3i::csv_to_parquet(opts);

    ASSERT_EQ(rep.column_names.size(), 2u);
    EXPECT_EQ(rep.column_names[0], "a");
    EXPECT_EQ(rep.column_names[1], "b");

    auto table = read_parquet(opts.output_path);
    ASSERT_EQ(table->num_columns(), 2);
    auto a = std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0));
    auto b = std::static_pointer_cast<arrow::Int64Array>(table->column(1)->chunk(0));
    EXPECT_EQ(a->Value(0), 1);
    EXPECT_EQ(a->Value(1), 3);
    EXPECT_EQ(b->Value(0), 2);
    EXPECT_EQ(b->Value(1), 4);
}

// Writing onto an existing file requires the overwrite flag.
TEST(CsvToParquet, OverwriteGuard) {
    TempDir tmp;
    const auto csv = tmp / "in.csv";
    write_text(csv, "x\n1\n");

    a3i::CsvToParquetOptions opts;
    opts.input_path  = csv;
    opts.output_path = tmp / "out.parquet";
    opts.has_header  = true;
    opts.overwrite   = false;

    ASSERT_NO_THROW((void)a3i::csv_to_parquet(opts));
    EXPECT_THROW((void)a3i::csv_to_parquet(opts), std::runtime_error);

    opts.overwrite = true;
    EXPECT_NO_THROW((void)a3i::csv_to_parquet(opts));
}
