// $Id: TranslationOptionCollectionConfusionNet.cpp 147 2007-10-14 21:36:11Z hieu $

#include "TranslationOptionCollectionConfusionNet.h"
#include "ConfusionNet.h"
#include "DecodeStep.h"
#include "LanguageModel.h"
#include "PhraseDictionaryMemory.h"
#include "LMList.h"

/** constructor; just initialize the base class */
TranslationOptionCollectionConfusionNet::TranslationOptionCollectionConfusionNet(
											const ConfusionNet &input) 
: TranslationOptionCollection(input) {}

/* forcibly create translation option for a particular source word.
	* call the base class' ProcessOneUnknownWord() for each possible word in the confusion network 
	* at a particular source position
*/
void TranslationOptionCollectionConfusionNet::ProcessUnknownWord(		
											size_t decodeStepId
											, size_t sourcePos) 
{
	ConfusionNet const& source=dynamic_cast<ConfusionNet const&>(m_source);

	ConfusionNet::Column const& coll=source.GetColumn(sourcePos);
	for(ConfusionNet::Column::const_iterator i=coll.begin();i!=coll.end();++i)
		ProcessOneUnknownWord(decodeStepId, i->first,sourcePos);
		
}
