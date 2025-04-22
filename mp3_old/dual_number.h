#ifndef DUAL_NUMBER
#define DUAL_NUMBER

#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
class dual_num {
public:

  float val, dval;

  //Basic constructors implemented:
  dual_num(){
    val = 0.0; dval = 0.0;
  }
  dual_num(float v){
    val = v; dval = 0.0;
  }
  dual_num(float v, float dv){
    val = v; dval = dv;
  }

  //basic operator overloads required for 

  dual_num operator+(const dual_num &that) const {
    return dual_num(val + that.val, dval + that.dval);
  }
  dual_num operator-(const dual_num &that) const {
    return dual_num(val - that.val, dval - that.dval);
  }
  dual_num operator*(const dual_num &that) const {
    return dual_num(val * that.val, val * that.dval + dval * that.val);
  }

  //get val and dval value
  float value() const {
    return val; 
  }
  float dual() const {
    return dval; 
  }

  //extended: if using equal assignment:
  dual_num &operator=(const dual_num &that) {
    if (this != &that) {
      this->val = that.val;
      this->dval = that.dval;
    }
    return *this;
  }
};



//Basic functions on dual numbers you should implement
//sin
inline dual_num sin(const dual_num &x) {
  return dual_num(std::sin(x.val), std::cos(x.val) * x.dval);
}
//cos
inline dual_num cos(const dual_num &x) {
  return dual_num(std::cos(x.val), -std::sin(x.val) * x.dval);
}
//exp
inline dual_num exp(const dual_num &x) {
  return dual_num(std::exp(x.val), std::exp(x.val) * x.dval);
}
//ln
inline dual_num ln(const dual_num &x) {
  if (x.val <= 0) {
     throw std::invalid_argument("ln(x) returned NaN\n");
    return dual_num(NAN, NAN);
  }
  return dual_num(std::log(x.val), (1 / x.val) * x.dval);
}
//relu
inline dual_num relu(const dual_num &x) {
  return dual_num(std::max(0.0f, x.val), (x.val > 0) ? x.dval : 0);
}
//sigmoid
inline dual_num sigmoid(const dual_num &x) {
  float func_val = 1 / (1 + std::exp(-x.val));
  return dual_num(func_val, func_val * (1 - func_val) * x.dval);
}
//tanh
inline dual_num tanh(const dual_num &x) {
  float func_val = std::tanh(x.val);
  return dual_num(func_val, (1 - func_val * func_val) * x.dval);
}

//the dual_vector class:
class dual_vector {
private:
  std::vector<dual_num> data;

public:
  // Constructors
  dual_vector() = default;
  dual_vector(size_t size) : data(size) {}
  dual_vector(const std::vector<dual_num>& vec) : data(vec) {}

  // Accessors
  size_t size() const { return data.size(); }

  dual_num& operator[](size_t i) { return data[i]; }
  const dual_num& operator[](size_t i) const { return data[i]; }

  std::vector<dual_num> as_std_vector() const { return data; }


  // Elementwise binary ops
  dual_vector operator+(const dual_vector& other) const {
    return elementwise_binary(other, std::plus<dual_num>());
  }

  dual_vector operator-(const dual_vector& other) const {
    return elementwise_binary(other, std::minus<dual_num>());
  }

  dual_vector operator*(const dual_vector& other) const {
    return elementwise_binary(other, std::multiplies<dual_num>());
  }

  // Elementwise unary ops
  friend dual_vector sin(const dual_vector& x) {
    return x.elementwise_unary(static_cast<dual_num(*)(const dual_num&)>(&::sin));
  }

  friend dual_vector cos(const dual_vector& x) {
    return x.elementwise_unary(static_cast<dual_num(*)(const dual_num&)>(&::cos));
  }

  friend dual_vector exp(const dual_vector& x) {
    return x.elementwise_unary(static_cast<dual_num(*)(const dual_num&)>(&::exp));
  }

  friend dual_vector ln(const dual_vector& x) {
    return x.elementwise_unary(::ln);
  }

  friend dual_vector relu(const dual_vector& x) {
    return x.elementwise_unary(::relu);
  }

  friend dual_vector sigmoid(const dual_vector& x) {
    return x.elementwise_unary(::sigmoid);
  }

  friend dual_vector tanh(const dual_vector& x) {
    return x.elementwise_unary(static_cast<dual_num(*)(const dual_num&)>(&::tanh));
  }

private:
  //helper function:
  dual_vector elementwise_binary(const dual_vector& other,
    std::function<dual_num(const dual_num&, const dual_num&)> op) const {
    if (data.size() != other.data.size()) {
      throw std::invalid_argument("dual_vector size mismatch");
    }
    dual_vector result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      result[i] = op(data[i], other.data[i]);
    }
    return result;
  }

  dual_vector elementwise_unary(std::function<dual_num(const dual_num&)> op) const {
    dual_vector result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      result[i] = op(data[i]);
    }
    return result;
  }
};





////////////////////////////////
#endif