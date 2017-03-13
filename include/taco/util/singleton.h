/// Define a singleton design pattern
#define SINGLETON(NAME)                     \
  public: 									\
	static NAME& get##NAME() { 				\
	  static NAME NAME_singleton;			\
	  return NAME_singleton;				\
  	  };									\
  	NAME(NAME const&) = delete; 			\
  	void operator=(NAME const&) = delete; 	\
  private: 									\
  	NAME() {};
