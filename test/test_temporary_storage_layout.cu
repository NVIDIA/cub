#include "cub/util_device.cuh"
#include "test_util.h"

#include <memory>

template <int Items>
std::size_t GetTemporaryStorageSize(std::size_t (&sizes)[Items])
{
  void *pointers[Items]{};
  std::size_t temp_storage_bytes{};
  cub::AliasTemporaries(nullptr, temp_storage_bytes, pointers, sizes);
  return temp_storage_bytes;
}

std::size_t GetActualZero()
{
  std::size_t sizes[1] {};

  return GetTemporaryStorageSize(sizes);
}

template <int StorageSlots>
void TestEmptyStorage()
{
  cub::TemporaryStorage::Layout<StorageSlots> temporary_storage;
  AssertEquals(temporary_storage.GetSize(), GetActualZero());
}

template <int StorageSlots>
void TestPartiallyFilledStorage()
{
  using target_type = std::uint64_t;
  constexpr std::size_t target_elements = 42;
  constexpr std::size_t full_slot_elements = target_elements * sizeof(target_type);
  constexpr std::size_t empty_slot_elements {};

  cub::TemporaryStorage::Layout<StorageSlots> temporary_storage;

  std::unique_ptr<cub::TemporaryStorage::Array<target_type>> arrays[StorageSlots];
  std::size_t sizes[StorageSlots] {};

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    auto slot = temporary_storage.GetSlot(slot_id);

    const std::size_t elements = slot_id % 2 == 0
                               ? full_slot_elements
                               : empty_slot_elements;
    sizes[slot_id] = elements * sizeof(target_type);
    arrays[slot_id].reset(new cub::TemporaryStorage::Array<target_type>(
      slot->template GetAlias<target_type>(elements)));
  }

  const std::size_t temp_storage_bytes = temporary_storage.GetSize();

  std::unique_ptr<std::uint8_t[]> temp_storage(
    new std::uint8_t[temp_storage_bytes]);

  temporary_storage.MapToBuffer(temp_storage.get(),
                                temp_storage_bytes);

  AssertEquals(temp_storage_bytes, GetTemporaryStorageSize(sizes));

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    if (slot_id % 2 == 0)
    {
      AssertTrue(arrays[slot_id]->Get() != nullptr);
    }
    else
    {
      AssertTrue(arrays[slot_id]->Get() == nullptr);
    }
  }
}

template <int StorageSlots>
void TestGrow()
{
  using target_type = std::uint64_t;
  constexpr std::size_t target_elements_number = 42;

  cub::TemporaryStorage::Layout<StorageSlots> preset_layout;
  std::unique_ptr<cub::TemporaryStorage::Array<target_type>> preset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    preset_arrays[slot_id].reset(
        new cub::TemporaryStorage::Array<target_type>(
            preset_layout.GetSlot(slot_id)->template GetAlias<target_type>(
              target_elements_number)));
  }

  cub::TemporaryStorage::Layout<StorageSlots> postset_layout;
  std::unique_ptr<cub::TemporaryStorage::Array<target_type>> postset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    postset_arrays[slot_id].reset(
        new cub::TemporaryStorage::Array<target_type>(
            postset_layout.GetSlot(slot_id)->template GetAlias<target_type>()));
    postset_arrays[slot_id]->Grow(target_elements_number);
  }

  AssertEquals(preset_layout.GetSize(), postset_layout.GetSize());

  const std::size_t tmp_storage_bytes = preset_layout.GetSize();
  std::unique_ptr<std::uint8_t[]> temp_storage(
      new std::uint8_t[tmp_storage_bytes]);

  preset_layout.MapToBuffer(temp_storage.get(), tmp_storage_bytes);
  postset_layout.MapToBuffer(temp_storage.get(), tmp_storage_bytes);

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    AssertEquals(postset_arrays[slot_id]->Get(), preset_arrays[slot_id]->Get());
  }
}

template <int StorageSlots>
void TestDoubleGrow()
{
  using target_type = std::uint64_t;
  constexpr std::size_t target_elements_number = 42;

  cub::TemporaryStorage::Layout<StorageSlots> preset_layout;
  std::unique_ptr<cub::TemporaryStorage::Array<target_type>> preset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    preset_arrays[slot_id].reset(
        new cub::TemporaryStorage::Array<target_type>(
            preset_layout.GetSlot(slot_id)->template GetAlias<target_type>(
                2 * target_elements_number)));
  }

  cub::TemporaryStorage::Layout<StorageSlots> postset_layout;
  std::unique_ptr<cub::TemporaryStorage::Array<target_type>> postset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    postset_arrays[slot_id].reset(
        new cub::TemporaryStorage::Array<target_type>(
            postset_layout.GetSlot(slot_id)->template GetAlias<target_type>(target_elements_number)));
    postset_arrays[slot_id]->Grow(2 * target_elements_number);
  }

  AssertEquals(preset_layout.GetSize(), postset_layout.GetSize());

  const std::size_t tmp_storage_bytes = preset_layout.GetSize();
  std::unique_ptr<std::uint8_t[]> temp_storage(
      new std::uint8_t[tmp_storage_bytes]);

  preset_layout.MapToBuffer(temp_storage.get(), tmp_storage_bytes);
  postset_layout.MapToBuffer(temp_storage.get(), tmp_storage_bytes);

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    AssertEquals(postset_arrays[slot_id]->Get(), preset_arrays[slot_id]->Get());
  }
}

template <int StorageSlots>
void Test()
{
  TestEmptyStorage<StorageSlots>();
  TestPartiallyFilledStorage<StorageSlots>();
  TestGrow<StorageSlots>();
  TestDoubleGrow<StorageSlots>();
}

int main()
{
  Test<1>();
  Test<4>();
  Test<42>();
}
