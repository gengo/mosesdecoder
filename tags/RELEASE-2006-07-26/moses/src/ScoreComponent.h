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

#pragma once

#include <iostream>
#include <vector>
#include <assert.h>
#include "TypeDef.h"

//typedef float ScoreComponent[NUM_PHRASE_SCORES];

class Dictionary;

//! TODO This should really be named TranslationCostBreakdown
class ScoreComponent
{
protected:
	const Dictionary *m_dictionary;
	std::vector<float> m_scoreComponent;

public:
	typedef std::vector<float>::iterator iterator;
	typedef std::vector<float>::const_iterator const_iterator;
	
	ScoreComponent();
	ScoreComponent(const Dictionary *dictionary);
	/**
	 * Deep copy
	 */
	ScoreComponent(const ScoreComponent &copy);
	~ScoreComponent() {}

	inline const Dictionary *GetDictionary() const
	{
		return m_dictionary;
	}

	size_t GetNumScoreComponents() const;

	float operator[](size_t index) const
	{
		return m_scoreComponent[index];
	}
	float &operator[](size_t index)
	{
		return m_scoreComponent[index];
	}
	iterator begin() {return m_scoreComponent.begin();}
	const_iterator begin() const {return m_scoreComponent.begin();}
	iterator end() {return m_scoreComponent.end();}
	const_iterator end() const {return m_scoreComponent.end();}
	size_t size() const {return m_scoreComponent.size();}
	
	void Reset();

	void PlusEquals(const ScoreComponent &source)
	{
		const size_t size = GetNumScoreComponents();
		assert(size == source.GetNumScoreComponents());

		for (size_t i = 0 ; i < size ; i++)
		{
			m_scoreComponent[i] += source[i];
		}
	}
};

std::ostream& operator<<(std::ostream &out, const ScoreComponent &scoreComponent);