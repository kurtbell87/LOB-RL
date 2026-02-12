#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "interfaces/features/IFeature.hpp"

/**
 * @file FeatureRegistry.hpp
 * @brief A global registry that maps "feature type name" -> factory function,
 *        enabling dynamic creation of features by string name.
 *
 * The registry also requires that created features implement:
 *     virtual void SetParam(const std::string& key, const std::string& val);
 * so we can pass user config (from Python) without enumerating them in code.
 *
 * Usage Steps:
 *   1) For each feature class (e.g. MyFeature) that wants to be instantiable
 *      by name, define or include the macro:
 *         REGISTER_FEATURE("MyFeature", MyFeature)
 *      in some .cpp file. That macro calls FeatureRegistry::RegisterFactory(...)
 *      with a unique string name and a lambda that does `std::make_shared<MyFeature>()`.
 *
 *   2) The Python side can now call: create_feature({"type": "MyFeature", ...})
 *      which calls FeatureRegistry::Instance().Create("MyFeature").
 *
 *   3) The registry calls the feature constructor, then we call SetParam(key,val)
 *      for each user-supplied config, so the feature can parse them as it likes.
 *
 *   4) No big chain of if(...) or switch(...) is required in Python or C++!
 */

namespace constellation {
namespace modules {
namespace features {

class FeatureRegistry {
public:
  /**
   * @brief Returns the singleton instance of the registry.
   */
  static FeatureRegistry& Instance();

  /**
   * @brief Factory function type: returns a fresh IFeature pointer.
   */
  using CreateFunc = std::function<std::shared_ptr<constellation::interfaces::features::IFeature>()>;

  /**
   * @brief Register a new feature type name => creation function.
   * @param typeName  e.g. "MidPriceFeature"
   * @param func      a lambda that returns make_shared<ThatFeatureClass>()
   * @throws std::runtime_error if typeName is already registered
   */
  void RegisterFactory(const std::string& typeName, const CreateFunc& func) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(typeName);
    if (it != factories_.end()) {
      throw std::runtime_error("FeatureRegistry: type '" + typeName + "' already registered!");
    }
    factories_[typeName] = func;
  }

  /**
   * @brief Create a feature by name. 
   * @throws std::runtime_error if name is unknown
   */
  std::shared_ptr<constellation::interfaces::features::IFeature> Create(const std::string& typeName) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(typeName);
    if (it == factories_.end()) {
      throw std::runtime_error("FeatureRegistry: unknown feature type '" + typeName + "'");
    }
    return it->second(); // call creation lambda
  }

private:
  FeatureRegistry() = default;
  ~FeatureRegistry() = default;
  FeatureRegistry(const FeatureRegistry&) = delete;
  FeatureRegistry& operator=(const FeatureRegistry&) = delete;

  std::mutex mutex_;
  std::unordered_map<std::string, CreateFunc> factories_;
};

/**
 * @brief Each feature .cpp can declare a static registrar with:
 *   REGISTER_FEATURE("MidPriceFeature", MidPriceFeature);
 * This macro expands to a small static struct that registers in the constructor.
 */
#define REGISTER_FEATURE(TYPE_NAME_STRING, FEATURE_CLASS)                        \
  static struct FEATURE_CLASS##_AutoReg {                                        \
    FEATURE_CLASS##_AutoReg() {                                                  \
      /* Use fully qualified name here: */                                       \
      ::constellation::modules::features::FeatureRegistry::Instance()            \
        .RegisterFactory(                                                        \
          TYPE_NAME_STRING,                                                      \
          [](){ return std::make_shared<FEATURE_CLASS>(); }                      \
        );                                                                       \
    }                                                                            \
  } g_##FEATURE_CLASS##_AutoReg;

} // end namespace features
} // end namespace modules
} // end namespace constellation
