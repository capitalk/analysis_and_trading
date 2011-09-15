/* 
 * File:   MarketPhases.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on May 21, 2010, 5:47 PM
 */

#ifndef _TemplateHelpers_HXX
#define	_TemplateHelpers_HXX

namespace common
{

template <class T>
struct IdentityExtractor
{

   T & operator()(T & x) const
   {
      return x;
   }

   const T & operator()(const T & x) const
   {
      return x;
   }
};

template <class Amount, class Object = Amount, class Extractor = IdentityExtractor<Object> >
struct IncrementBy
{
   Amount _by;
   Extractor _ex;

   IncrementBy(const Amount & by) : _by(by)
   {

   }

   void operator()(Object & x) const
   {
      _ex(x) += _by;
   }
};

}

#endif	/* _TemplateHelpers_HXX */

