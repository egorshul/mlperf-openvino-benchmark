// convert_decoder_static — transform a stateful Whisper decoder model so that
// KV-cache tensors become explicit inputs/outputs with static shapes suitable
// for accelerators that cannot handle dynamic state Variables.
//
// Blueprint: openvinotoolkit/openvino.genai  src/cpp/src/whisper/pipeline_static.cpp
//
// Transformations applied:
//   1. StatefulToStateless — ReadValue → Parameter, Assign → Result.
//   2. redirect_new_kv_to_output — for every self-attention "present.*decoder*"
//      Result, replace the Concat(past,new) output with just `new` so the
//      model emits only freshly-computed KV of shape [1,heads,seq_len,head_dim].
//   3. Save the transformed model.
//
// The resulting model can be reshaped to fully-static shapes in the Python
// runtime and compiled on any device.
//
// Usage:
//   convert_decoder_static <model_dir>
//
//   Reads   <model_dir>/decoder_with_past_model.xml
//   Writes  <model_dir>/decoder_static.xml  (+.bin)

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/result.hpp>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string var_id_of(const std::shared_ptr<ov::Node>& node) {
    auto attrs = node->get_attributes();
    auto it = attrs.find("variable_id");
    if (it != attrs.end())
        return it->second.as<std::string>();
    return {};
}

// Walk backwards from `start` towards model inputs looking for the first
// Concat node.  Returns nullptr if not found within `max_depth` hops.
static std::shared_ptr<ov::op::v0::Concat>
find_concat_backwards(const ov::Output<ov::Node>& start, int max_depth = 8) {
    auto out = start;
    for (int i = 0; i < max_depth; ++i) {
        auto node = out.get_node_shared_ptr();
        if (auto cat = std::dynamic_pointer_cast<ov::op::v0::Concat>(node))
            return cat;
        if (node->get_input_size() == 0)
            break;
        out = node->input(0).get_source_output();
    }
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — StatefulToStateless (manual implementation)
//
// ReadValue(init) → consumers   ⟹   Parameter → consumers
// Assign(value)                 ⟹   Result(value)
// ─────────────────────────────────────────────────────────────────────────────

static void stateful_to_stateless(std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_params;
    std::vector<std::shared_ptr<ov::op::v0::Result>>    new_results;
    std::unordered_set<std::shared_ptr<ov::Node>>        to_remove;

    // --- ReadValue → Parameter -------------------------------------------
    for (auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() != std::string("ReadValue"))
            continue;

        auto vid = var_id_of(node);
        auto out_shape = node->get_output_partial_shape(0);
        auto out_type  = node->get_output_element_type(0);

        // Create a replacement Parameter with the same shape/type.
        auto param = std::make_shared<ov::op::v0::Parameter>(out_type, out_shape);
        std::string param_name = vid.empty()
            ? std::string("past_kv_") + std::to_string(new_params.size())
            : vid;
        param->set_friendly_name(param_name);
        param->get_output_tensor(0).set_names({param_name});

        // Redirect every consumer of ReadValue to the new Parameter.
        node->output(0).replace(param->output(0));

        new_params.push_back(param);
        to_remove.insert(node);

        std::cout << "  ReadValue '" << vid << "' → Parameter '"
                  << param_name << "'  " << out_shape << "\n";
    }

    // --- Assign → Result -------------------------------------------------
    for (auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() != std::string("Assign"))
            continue;

        auto vid = var_id_of(node);
        auto src = node->input(0).get_source_output();

        std::string result_name = vid.empty()
            ? std::string("present_kv_") + std::to_string(new_results.size())
            : std::regex_replace(vid, std::regex("^past_key_values"),
                                 "present");
        // If vid doesn't start with "past_key_values", just prefix with
        // "present.":
        if (result_name == vid)
            result_name = "present." + vid;

        auto result = std::make_shared<ov::op::v0::Result>(src);
        result->set_friendly_name(result_name);
        result->get_output_tensor(0).set_names({result_name});

        new_results.push_back(result);
        to_remove.insert(node);

        std::cout << "  Assign    '" << vid << "' → Result '"
                  << result_name << "'\n";
    }

    // --- Wire new Parameters / Results into the model --------------------
    auto params  = model->get_parameters();
    auto results = model->get_results();
    auto sinks   = model->get_sinks();

    params.insert(params.end(), new_params.begin(), new_params.end());
    results.insert(results.end(), new_results.begin(), new_results.end());

    // Remove Assign sinks.
    ov::SinkVector remaining_sinks;
    for (auto& s : sinks) {
        if (to_remove.count(s) == 0)
            remaining_sinks.push_back(s);
    }

    model = std::make_shared<ov::Model>(results, remaining_sinks, params,
                                        model->get_friendly_name());
    std::cout << "  StatefulToStateless: +" << new_params.size()
              << " params, +" << new_results.size() << " results\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 — redirect_new_kv_to_output
//
// For each Result that comes from a self-attention KV Concat, replace
//     Result ← Concat(past, new)
// with
//     Result ← new
//
// This makes the output shape [1, heads, seq_len, head_dim] (small, static)
// instead of the growing [1, heads, past+seq_len, head_dim].
// The Concat itself stays in the graph — SDPA still uses it for attention.
// ─────────────────────────────────────────────────────────────────────────────

static int redirect_new_kv_to_output(std::shared_ptr<ov::Model>& model) {
    int redirected = 0;

    for (auto& result : model->get_results()) {
        auto name = result->get_friendly_name();

        // Only redirect self-attention decoder KV outputs.
        // Cross-attention ("encoder") outputs stay as-is.
        bool is_decoder_kv =
            name.find("decoder") != std::string::npos &&
            (name.find("present") != std::string::npos ||
             name.find("present_kv") != std::string::npos);
        if (!is_decoder_kv)
            continue;

        // Walk backwards to find the Concat node.
        auto concat = find_concat_backwards(
            result->input(0).get_source_output());
        if (!concat) {
            std::cerr << "  WARNING: no Concat found for '" << name
                      << "', skipping redirect\n";
            continue;
        }

        // Concat(past_kv, new_kv, axis=2).  Input[1] is the new KV.
        auto new_kv = concat->input(1).get_source_output();
        result->input(0).replace_source_output(new_kv);
        ++redirected;

        std::cout << "  Redirect '" << name << "' → new_kv "
                  << new_kv.get_partial_shape() << "\n";
    }

    return redirected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: convert_decoder_static <model_dir>\n"
                  << "\n"
                  << "Reads   <model_dir>/decoder_with_past_model.xml\n"
                  << "Writes  <model_dir>/decoder_static.xml  (+.bin)\n";
        return 1;
    }

    fs::path model_dir(argv[1]);
    fs::path src_xml = model_dir / "decoder_with_past_model.xml";
    fs::path dst_xml = model_dir / "decoder_static.xml";

    if (!fs::exists(src_xml)) {
        std::cerr << "ERROR: " << src_xml << " not found\n";
        return 1;
    }

    std::cout << "=== convert_decoder_static ===\n";
    std::cout << "Source: " << src_xml << "\n";

    // ── Load model ──────────────────────────────────────────────────────
    ov::Core core;
    auto model = core.read_model(src_xml.string());

    std::cout << "\nOriginal model:\n"
              << "  Inputs:  " << model->get_parameters().size() << "\n"
              << "  Outputs: " << model->get_results().size() << "\n";

    size_t n_vars = 0;
    try { n_vars = model->get_variables().size(); } catch (...) {}
    std::cout << "  Variables (state): " << n_vars << "\n";

    if (n_vars == 0) {
        std::cerr << "ERROR: model has no state Variables — nothing to convert\n";
        return 1;
    }

    // ── Step 1: StatefulToStateless ─────────────────────────────────────
    std::cout << "\n[Step 1] StatefulToStateless\n";
    stateful_to_stateless(model);

    // ── Step 2: redirect self-attention KV outputs ──────────────────────
    std::cout << "\n[Step 2] Redirect self-attention KV outputs\n";
    int n_redirected = redirect_new_kv_to_output(model);
    std::cout << "  Redirected " << n_redirected << " outputs\n";

    // ── Summary ─────────────────────────────────────────────────────────
    std::cout << "\nTransformed model:\n"
              << "  Inputs:  " << model->get_parameters().size() << "\n"
              << "  Outputs: " << model->get_results().size() << "\n";

    std::cout << "\n  Inputs:\n";
    for (auto& p : model->get_parameters()) {
        std::cout << "    " << p->get_friendly_name()
                  << " : " << p->get_output_partial_shape(0)
                  << " " << p->get_output_element_type(0) << "\n";
    }
    std::cout << "  Outputs:\n";
    for (auto& r : model->get_results()) {
        std::cout << "    " << r->get_friendly_name()
                  << " : " << r->input(0).get_partial_shape()
                  << "\n";
    }

    // ── Save ────────────────────────────────────────────────────────────
    std::cout << "\nSaving to " << dst_xml << " ...\n";
    ov::save_model(model, dst_xml.string());
    std::cout << "Done.\n";

    return 0;
}
