#include "test.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/properties.h"

using namespace taco;

TEST(properties, annihilator) {
  Literal z(0);
  Annihilator a(nullptr);
  Annihilator zero(z);
  ASSERT_FALSE(a.defined());
  ASSERT_TRUE(zero.defined());

  ASSERT_EQ(zero.annihilator(), z);
  ASSERT_TRUE(equals(zero.annihilator(), z));

  ASSERT_TRUE(zero.equals(Annihilator(Literal(0))));
  ASSERT_FALSE(zero.equals(a));
}

TEST(properties, identity) {
  Literal z(0);
  Identity a(nullptr);
  Identity zero(z);
  ASSERT_FALSE(a.defined());
  ASSERT_TRUE(zero.defined());

  ASSERT_EQ(zero.identity(), z);
  ASSERT_TRUE(equals(zero.identity(), Literal(0)));

  ASSERT_TRUE(zero.equals(Identity(Literal(0))));
  ASSERT_FALSE(zero.equals(a));
}

TEST(properties, associative) {
  Associative a;
  Associative undef(nullptr);

  ASSERT_TRUE(a.equals(a));
  ASSERT_FALSE(a.equals(undef));
  ASSERT_TRUE(a.defined());
  ASSERT_FALSE(undef.defined());
}

TEST(properties, commutative) {
  Commutative com;
  Commutative specific({0, 1});
  Commutative specific2({1, 2});
  Commutative undef(nullptr);

  ASSERT_TRUE(specific.defined());
  ASSERT_TRUE(com.defined());
  ASSERT_FALSE(undef.defined());

  ASSERT_NE(specific.ordering(), specific2.ordering());
  ASSERT_EQ(specific.ordering(), std::vector<int>({0, 1}));
  ASSERT_TRUE(specific.equals(specific));
}

TEST(properties, property_conversion) {
  Property annh = Annihilator(10);
  Property identity = Identity(40);
  Property assoc = Associative();
  Property com = Commutative({0,1,2});

  ASSERT_TRUE(isa<Annihilator>(annh));
  ASSERT_FALSE(isa<Identity>(annh));
  Annihilator a = to<Annihilator>(annh);
  ASSERT_TRUE(equals(a.annihilator(), Literal(10)));

  ASSERT_TRUE(isa<Identity>(identity));
  ASSERT_FALSE(isa<Annihilator>(identity));
  Identity idnty = to<Identity>(identity);
  ASSERT_TRUE(equals(idnty.identity(), Literal(40)));

  ASSERT_TRUE(isa<Associative>(assoc));
  ASSERT_FALSE(isa<Commutative>(assoc));
  Associative assc = to<Associative>(assoc);
  ASSERT_TRUE(assc.defined());

  ASSERT_TRUE(isa<Commutative>(com));
  ASSERT_FALSE(isa<Associative>(com));
  Commutative comm = to<Commutative>(com);
  ASSERT_EQ(comm.ordering(), std::vector<int>({0,1,2}));
}

TEST(properties, findProperty) {
  Annihilator a(10);
  Identity i(10);
  Associative as;
  Commutative c({0, 1});

  std::vector<Property> properties({a, i, as, c});
  ASSERT_TRUE(a.equals(findProperty<Annihilator>(properties)));
  ASSERT_TRUE(i.equals(findProperty<Identity>(properties)));
  ASSERT_TRUE(as.equals(findProperty<Associative>(properties)));
  ASSERT_TRUE(c.equals(findProperty<Commutative>(properties)));

  std::vector<Property> partialProperties({a, c});
  ASSERT_FALSE(i.equals(findProperty<Identity>(partialProperties)));
  ASSERT_FALSE(as.equals(findProperty<Associative>(partialProperties)));

  ASSERT_FALSE(findProperty<Identity>(partialProperties).defined());
  ASSERT_FALSE(findProperty<Associative>(partialProperties).defined());

  ASSERT_TRUE(properties[0].equals(Annihilator(10)));
  ASSERT_FALSE(properties[0].equals(i));
}