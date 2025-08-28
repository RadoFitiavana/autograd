#pragma once

#include <type_traits>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <string>
#include <valarray>
#include <memory>

namespace autoDiff::utils
{
	template<typename T>
	class Node final
	{
		std::function<T(const std::vector<T>&)> _fun;
		std::function<std::vector<T>(const std::vector<T>&)> _grad_fun;
	public:
		explicit Node(const std::function<T(const std::vector<T>&)>& fun,
			          const std::function<std::vector<T>(const std::vector<T>&)>& grad_fun);
		~Node() = default;
		Node() = default;

		Node(const Node&) = delete;
		Node(Node&&) = delete;

		Node& operator=(const Node&) = delete;
		Node& operator=(Node&&) = delete;

		void make_fun(const std::function<T(const std::vector<T>&)>& fun);
		void make_grad_fun(const std::function<std::vector<T>(const std::vector<T>&)>& grad_fun);

		std::vector<T> grad(const std::vector<T>& in);

		T operator()(const std::vector<T>& in);
	};

	template<typename T>
	Node<T>::Node(const std::function<T(const std::vector<T>&)>& fun,
		const std::function<std::vector<T>(const std::vector<T>&)>& grad_fun)
		: _fun(fun),
		  _grad_fun(grad_fun)
	{
	}

	template <typename T>
	T Node<T>::operator()(const std::vector<T>& in)
	{
		return _fun(in);
	}

	template<typename T>
	void Node<T>::make_fun(const std::function<T(const std::vector<T>&)>& fun)
	{
		_fun = fun;
	}

	template<typename T>
	void Node<T>::make_grad_fun(const std::function<std::vector<T>(const std::vector<T>&)>& grad_fun)
	{
		_grad_fun = grad_fun;
	}

	template <typename T>
	std::vector<T> Node<T>::grad(const std::vector<T>& in)
	{
		return _grad_fun(in);
	}

}

namespace autoDiff::cg
{
	template<typename T>
	class Var
	{
		static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
			         "Allowed types are float or double");

		bool _visited;
		std::vector<std::shared_ptr<Var<T>>> _parents;
		std::vector<std::shared_ptr<Var<T>>> _children;
		std::vector<T> _grad;

	public:
		T _value;
		bool _requires_grad;
		utils::Node<T> _fun;
		explicit Var(const T& value);
		explicit Var(const T& value, const bool& requires_grad);
		Var() = delete;
		~Var() = default;

		Var(const Var&) = delete;
		Var(Var&&) = delete;
		Var& operator=(const Var&) = delete;
		Var& operator=(Var&&) = delete;

		std::vector<T> extract();

		void setParents(const std::vector<std::shared_ptr<Var>>& parents);

		size_t appendChild(const std::shared_ptr<Var>& child)
		{
			_children.push_back(child);
			return _children.size();
		}

		bool& visited() { return _visited; }

		std::vector<std::shared_ptr<Var>>& parents() { return _parents; }

		std::vector<std::shared_ptr<Var>>& children() { return _children; }

		std::vector<T>& grad() { return _grad; }
	};

	template<typename T>
	Var<T>::Var(const T& value)
	{
		_value = value;
		_visited = false;
		_requires_grad = true;
	}

	template<typename T>
	Var<T>::Var(const T& value, const bool& requires_grad)
	{
		_value = value;
		_visited = false;
		_requires_grad = requires_grad;
	}

	template<typename T>
	std::vector<T> cg::Var<T>::extract()
	{
		std::vector<T> in;
		for (const std::shared_ptr<Var<T>>& parent : _parents)
		{
			in.push_back(parent->_value);
		}
		return in;
	}

	template<typename T>
	void Var<T>::setParents(const std::vector<std::shared_ptr<Var>>& parents)
	{
		_parents = parents;
	}

	template<typename T>
	std::shared_ptr<Var<T>> make_var(const T& value)
	{
		return std::make_shared<Var<T>>(value);
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator+(const std::shared_ptr<Var<T>>& in1, const std::shared_ptr<Var<T>>& in2)
	{
		std::shared_ptr<Var<T>> out = make_var(in1->_value + in2->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] + in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{1, 1};
			}
		);

		out->setParents({ in1, in2 });
		in1->appendChild(out);
		in2->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator-(const std::shared_ptr<Var<T>>& in1, const std::shared_ptr<Var<T>>& in2)
	{
		std::shared_ptr<Var<T>> out = make_var(in1->_value - in2->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] - in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{1, -1};
			}
		);

		out->setParents({ in1, in2 });
		in1->appendChild(out);
		in2->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator*(const std::shared_ptr<Var<T>>& in1, const std::shared_ptr<Var<T>>& in2)
	{
		std::shared_ptr<Var<T>> out = make_var(in1->_value * in2->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] * in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{in[1], in[0]};
			}
		);

		out->setParents({ in1, in2 });
		in1->appendChild(out);
		in2->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator/(const std::shared_ptr<Var<T>>& in1, const std::shared_ptr<Var<T>>& in2)
	{
		std::shared_ptr<Var<T>> out = make_var(in1->_value / in2->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] / in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{1/in[1], -(in[0]/std::pow(in[1],2))};
			}
		);

		out->setParents({ in1, in2 });
		in1->appendChild(out);
		in2->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator+(const T& constant, const std::shared_ptr<Var<T>>& var)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(scalar->_value + var->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] + in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return { 0.0 , 1 };
			}
		);

		out->setParents({ scalar, var });
		scalar->appendChild(out);
		var->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator+(const std::shared_ptr<Var<T>>& var, const T& constant)
	{
		return constant + var;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator*(const T& constant, const std::shared_ptr<Var<T>>& var)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(scalar->_value * var->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] * in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return { 0.0 , in[0] };
			}
		);

		out->setParents({ scalar, var });
		scalar->appendChild(out);
		var->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator*(const std::shared_ptr<Var<T>>& var, const T& constant)
	{
		return constant * var;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator-(const T& constant, const std::shared_ptr<Var<T>>& var)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(scalar->_value - var->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] - in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return { 0.0 , -1 };
			}
		);

		out->setParents({ scalar, var });
		scalar->appendChild(out);
		var->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator-(const std::shared_ptr<Var<T>>& var, const T& constant)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(var->_value - scalar->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] - in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return { 1 , 0.0 };
			}
		);

		out->setParents({ var , scalar });
		scalar->appendChild(out);
		var->appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator/(const T& constant, const std::shared_ptr<Var<T>>& var)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(scalar->_value / var->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] / in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{ 0.0 , -(in[0] / std::pow(in[1],2))};
			}
		);

		out->setParents({ scalar, var });
		scalar->appendChild(out);
		var.appendChild(out);

		return out;
	}

	template<typename T>
	std::shared_ptr<Var<T>> operator/(const std::shared_ptr<Var<T>>& var, const T& constant)
	{
		std::shared_ptr<Var<T>> scalar = make_var(constant);
		scalar->_requires_grad = false;
		std::shared_ptr<Var<T>> out = make_var(var->_value / scalar->_value);
		out->_fun.make_fun(
			[](const std::vector<T>& in) -> T
			{
				return in[0] / in[1];
			}
		);

		out->_fun.make_grad_fun(
			[](const std::vector<T>& in) -> std::vector<T>
			{
				return std::vector<T>{1 / in[1] , 0.0};
			}
		);

		out->setParents({ var , scalar });
		scalar->appendChild(out);
		var->appendChild(out);

		return out;
	}

}

namespace autoDiff::node
{
	template<typename T>
	std::function<std::shared_ptr<cg::Var<T>> (const std::vector<std::shared_ptr<cg::Var<T>>>&)> build_node(const std::function<T(const std::vector<T>&)>& fun,
	                                                                                                        const std::function<std::vector<T>(const std::vector<T>&)>& grad_fun)
	{
		return [=](const std::vector<std::shared_ptr<cg::Var<T>>>& varList) -> std::shared_ptr<cg::Var<T>>
		{
			std::vector<T> in = {};
			for (const std::shared_ptr<cg::Var<T>>& var : varList)
			{
				in.push_back(var->_value);
			}
				
			std::shared_ptr<cg::Var<T>> out = cg::make_var(fun(in));
			out->_fun.make_fun(fun);

			out->_fun.make_grad_fun(grad_fun);

			out->setParents(varList);
			for (const std::shared_ptr<cg::Var<T>>& var : varList)
			{
				var->appendChild(out);
			}

			return out;
		};
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> log(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
			{
				return std::log(in[0]);
			}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
			{
				return { 1 / in[0] };
			})
		);

		return fun({var});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> log(const std::shared_ptr<cg::Var<T>>& var, const T& base)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::log(in[0]) / std::log(in[1]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { 1 / (in[0] * std::log(in[1])) , 0.0 };
				})
		);

		return fun({var , cg::make_var(base)});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> exp(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::exp(in[0]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { std::exp(in[0])};
				})
		);

		return fun({var});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> sin(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::sin(in[0]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { std::cos(in[0]) };
				})
		);

		return fun({var});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> cos(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::cos(in[0]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { - std::sin(in[0]) };
				})
		);

		return fun({var});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> abs(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::abs(in[0]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					if (in[0] == 0.0)
					{
						throw std::runtime_error("abs function is not differentiable in 0");
					}
					 
					return in[0] > 0 ? std::vector<T>{ 1.0 } : std::vector<T>{ - 1.0 };
				})
		);

		return fun({var});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> sqr(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::pow(in[0], 2);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { 2.0 * in[0]};
				})
		);

		return fun(std::vector<std::shared_ptr<cg::Var<T>>>{var});
	}

	template<typename T>
		std::shared_ptr<cg::Var<T>> pow(const std::shared_ptr<cg::Var<T>>& var, const int& expo)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::pow(in[0], in[1]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					return { in[1] * std::pow(in[0], in[1]-1) , 0.0 };
				})
		);

		return fun({var , cg::make_var(static_cast<T>(expo))});
	}

	template<typename T>
	std::shared_ptr<cg::Var<T>> sqrt(const std::shared_ptr<cg::Var<T>>& var)
	{
		auto fun = build_node(
			std::function<T(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> T
				{
					return std::sqrt(in[0]);
				}),
			std::function<std::vector<T>(const std::vector<T>&)>(
				[](const std::vector<T>& in) -> std::vector<T>
				{
					if (in[0] == 0.0)
					{
						throw std::runtime_error("sqrt is not differentiable in 0");
					}
					return { 0.5 / std::sqrt(in[0]) };
				})
		);

		return fun({var});
	}
}

namespace autoDiff
{

	/******************** // alias name for Var<T>* // ****************************/
								template<typename T>
								using var = std::shared_ptr<cg::Var<T>>;
	/*******************************************************************************/

	template<typename T>
	class AutoFunction final
	{
		static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
			          "Allowed types are float or double");

		std::function<std::shared_ptr<cg::Var<T>> (const std::vector<std::shared_ptr<cg::Var<T>>>&)> _wrapper;
		size_t _input_dim;
		std::vector<std::shared_ptr<cg::Var<T>>> _varList;
		std::shared_ptr<cg::Var<T>> _out;
		std::vector<std::shared_ptr<cg::Var<T>>> _topo;

		void build_graph();
		void TopologicalSort(std::shared_ptr<cg::Var<T>>& node);
		void reset();
		void setVar_values(const std::vector<T>& var_values);
		std::vector<T> make_reverseMode();
		std::vector<T> make_forwardMode();

	public:
		explicit AutoFunction(size_t input_dim,
			                  const std::function<std::shared_ptr<cg::Var<T>>(const std::vector<std::shared_ptr<cg::Var<T>>>&)>& wrapper);
		~AutoFunction() = default;
		AutoFunction() = delete;
		AutoFunction(const AutoFunction&) = delete;
		AutoFunction(AutoFunction&&) = delete;
		AutoFunction& operator=(const AutoFunction&) = delete;
		AutoFunction& operator=(AutoFunction&&) = delete;
		T operator()(const std::vector<T>& var_values);

		void set_variable(const size_t& index, const T& value, const bool& requires_grad=true);

		std::vector<T> grad(const std::vector<T>& var_values, bool reverseMode=false);

	};

	template<typename T>
	void AutoFunction<T>::TopologicalSort(std::shared_ptr<cg::Var<T>>& node)
	{
		if (!node->visited())
		{
			node->visited() = true;
			for (std::shared_ptr<cg::Var<T>>& parent : node->parents())
			{
				TopologicalSort(parent);
			}
			_topo.push_back(node);
		}
	}

	template<typename T>
	void AutoFunction<T>::build_graph()
	{
		if (_out == nullptr)
		{
			_out = _wrapper(_varList);
			TopologicalSort(_out);
		}
	}

	template<typename T>
	AutoFunction<T>::AutoFunction(size_t input_dim,
		                          const std::function<std::shared_ptr<cg::Var<T>> (const std::vector<std::shared_ptr<cg::Var<T>>>&)>& wrapper)
				    : _wrapper(wrapper),
	                  _input_dim(input_dim)
	{
		_out = nullptr;
	}

	template<typename T>
	T AutoFunction<T>::operator()(const std::vector<T>& var_values)
	{
		if (var_values.size() != _input_dim)
		{
			std::string reason = "given values size is " + std::to_string(var_values.size()) + " but variables size is " + std::to_string(_varList.size());
			throw std::runtime_error("invalid variable values: " + reason);
		}
		setVar_values(var_values);
		build_graph();
		for (std::shared_ptr<cg::Var<T>>& node : _topo)
		{
			for (std::shared_ptr<cg::Var<T>>& child : node->children())
			{
				child->_value = child->_fun(child->extract());
			}
		}
		reset();
		return _out->_value;
	}

	template<typename T>
	void AutoFunction<T>::set_variable(const size_t& index, const T& value, const bool& requires_grad)
	{
		if (_varList.empty())
		{
			throw std::runtime_error("Input variables are missing");
		}
		if (index < 0 || index >= _input_dim)
		{
			throw std::runtime_error("index out of range");
		}

		_varList[index]->_value = value;
		_varList[index]->_requires_grad = requires_grad;
	}

	template <typename T>
	void AutoFunction<T>::setVar_values(const std::vector<T>& var_values)
	{
		if (_varList.empty())
		{
			for (const T& value : var_values)
			{
				_varList.push_back(cg::make_var(value));
			}
			return;
		}
		for (size_t i=0; i<_input_dim; i++)
		{
			_varList[i]->_value = var_values[i];
		}
	}

	template <typename T>
	void AutoFunction<T>::reset()
	{
		for (std::shared_ptr<cg::Var<T>>& node : _topo)
		{
			std::vector<T>().swap(node->grad());
			node->visited() = false;
		}
	}

	template<typename T>
	std::vector<T> AutoFunction<T>::grad(const std::vector<T>& var_values, bool reverseMode)
	{
		if (var_values.size() != _input_dim)
		{
			std::string reason = "given values size is " + std::to_string(var_values.size()) + " but variables size is " + std::to_string(_varList.size());
			throw std::runtime_error("invalid variable values: " + reason);
		}
		setVar_values(var_values);

		if (reverseMode)
		{
			return make_reverseMode();
		}

		return make_forwardMode();
	}

	template<typename T>
	std::vector<T> AutoFunction<T>::make_forwardMode()
	{
		build_graph();
		std::vector<T> tmp_grad;

		for (size_t i = 0; i < _input_dim; i++) {
			for (size_t j = 0; j < _input_dim; j++) {
				if (_varList[j]->_requires_grad) {
					if (_varList[j]->grad().size() <= i)
						_varList[j]->grad().resize(i + 1, 0);
					_varList[j]->grad()[i] = (i == j) ? 1 : 0;
				}
			}

			for (std::shared_ptr<cg::Var<T>>& node : _topo) {
				if (!node->_requires_grad || node->parents().empty())
					continue;

				std::vector<T> local_grad = node->_fun.grad(node->extract());

				if (node->grad().size() <= i)
					node->grad().resize(i + 1, 0);

				size_t index = 0;
				for (std::shared_ptr<cg::Var<T>>& parent : node->parents()) {
					if (!parent || !parent->_requires_grad) {
						++index;
						continue;
					}

					if (parent->grad().size() <= i)
						parent->grad().resize(i + 1, 0);

					if (index >= local_grad.size()) {
						throw std::runtime_error("Mismatch between parent size and grad fun output");
					}

					node->grad()[i] += parent->grad()[i] * local_grad[index];
					++index;
				}
			}

			tmp_grad.push_back(_out->grad()[i]);
		}

		reset();
		return tmp_grad;
	}

	template<typename T>
	std::vector<T> AutoFunction<T>::make_reverseMode()
	{
		build_graph();

		_out->grad().resize(1, 1.0);

		for (auto it = _topo.rbegin(); it != _topo.rend(); ++it) {
			std::shared_ptr<cg::Var<T>>& node = *it;

			if (!node->_requires_grad || node->parents().empty())
				continue;

			std::vector<T> local_grad = node->_fun.grad(node->extract());

			for (size_t i = 0; i < node->parents().size(); ++i) {
				std::shared_ptr<cg::Var<T>> parent = node->parents()[i];
				if (!parent->_requires_grad) continue;

				if (parent->grad().empty())
					parent->grad().resize(1, 0.0);

				if (node->grad().empty())
					node->grad().resize(1, 0.0);

				parent->grad()[0] += node->grad()[0] * local_grad[i];
			}
		}

		std::vector<T> tmp_grad;
		for (size_t i = 0; i < _input_dim; ++i) {
			if (_varList[i]->grad().empty())
				tmp_grad.push_back(0.0);
			else
				tmp_grad.push_back(_varList[i]->grad()[0]);
		}

		reset();
		return tmp_grad;
	}
}