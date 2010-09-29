// $Id$

/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2006 University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <cassert>
#include <cstring>
#include "lm/ngram.hh"

#include "LanguageModelKen.h"
#include "TypeDef.h"
#include "Util.h"
#include "FactorCollection.h"
#include "Phrase.h"
#include "InputFileStream.h"
#include "StaticData.h"

using namespace std;

namespace Moses
{

namespace {
struct KenLMState : public FFState {
  lm::ngram::State state;
  int Compare(const FFState &o) const {
    const KenLMState &other = static_cast<const KenLMState &>(o);
    if (state.valid_length_ < other.state.valid_length_) return -1;
    if (state.valid_length_ > other.state.valid_length_) return 1;
    return std::memcmp(state.history_, other.state.history_, sizeof(lm::WordIndex) * state.valid_length_);
  }
};
} // namespace

LanguageModelKen::LanguageModelKen(bool registerScore, ScoreIndexManager &scoreIndexManager)
:LanguageModelSingleFactor(registerScore, scoreIndexManager), m_ngram(NULL)
{
}

LanguageModelKen::~LanguageModelKen()
{
	delete m_ngram;
}

bool LanguageModelKen::Load(const std::string &filePath, 
			     FactorType factorType, 
			     size_t /*nGramOrder*/)
{
	m_ngram = new lm::ngram::Model(filePath.c_str());

	m_factorType  = factorType;
	m_nGramOrder  = m_ngram->Order();
	m_filePath    = filePath;

	FactorCollection &factorCollection = FactorCollection::Instance();
	m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
	m_sentenceStartArray[m_factorType] = m_sentenceStart;
	m_sentenceEnd = factorCollection.AddFactor(Output, m_factorType, EOS_);
	m_sentenceEndArray[m_factorType] = m_sentenceEnd;

  KenLMState *tmp = new KenLMState();
  tmp->state = m_ngram->NullContextState();
  m_emptyHypothesisState = tmp;
  tmp = new KenLMState();
  tmp->state = m_ngram->BeginSentenceState();
  m_beginSentenceState = tmp;
	return true;
}

	/* get score of n-gram. n-gram should not be bigger than m_nGramOrder
	 * Specific implementation can return State and len data to be used in hypothesis pruning
	 * \param contextFactor n-gram to be scored
	 * \param finalState state used by LM. Return arg
	 * \param len ???
	 */	
float LanguageModelKen::GetValueAndState(const vector<const Word*> &contextFactor, FFState &outState, unsigned int* len) const
{
	FactorType factorType = GetFactorType();
	size_t count = contextFactor.size();
	assert(count <= GetNGramOrder());
	if (count == 0)
	{
		static_cast<KenLMState&>(outState).state = m_ngram->NullContextState();
		return 0;
	}
	
	// set up context
	vector<lm::WordIndex> ngramId(count);
	for (size_t i = 0 ; i < count; i++)
	{
		const Factor *factor = contextFactor[i]->GetFactor(factorType);
		const string &word = factor->GetString();
		
		// TODO(hieuhoang1972): precompute this.   
		ngramId[count - 1 - i] = m_ngram->GetVocabulary().Index(word);
	}

  lm::FullScoreReturn ret(m_ngram->FullScoreForgotState(&*ngramId.begin() + 1, &*ngramId.end(), ngramId.front(), static_cast<KenLMState&>(outState).state));
	if (len)
	{
		*len = ret.ngram_length;
	}
	return TransformLMScore(ret.prob);
}

lm::WordIndex LanguageModelKen::GetLmID(const std::string &str) const {
	return m_ngram->GetVocabulary().Index(str);
}

}

