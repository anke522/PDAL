/******************************************************************************
 * Copyright (c) 2019, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 ****************************************************************************/

#include "IterativeClosestPoint.hpp"

#include <pdal/EigenUtils.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/util/Utils.hpp>

#include <Eigen/Dense>

#include <numeric>

namespace pdal
{

static StaticPluginInfo const s_info
{
    "filters.icp",
    "Iterative Closest Point (ICP) registration.",
    "http://pdal.io/stages/filters.icp.html"
};

CREATE_STATIC_STAGE(IterativeClosestPoint, s_info)

std::string IterativeClosestPoint::getName() const
{
    return s_info.name;
}

void IterativeClosestPoint::addArgs(ProgramArgs& args)
{
    args.add("max_iter", "Maximum number of iterations", m_max_iters, 100);
    // various convergence criteria and threshold need to be added
}

PointViewSet IterativeClosestPoint::run(PointViewPtr view)
{
    using namespace Dimension;

    PointViewSet viewSet;
    if (this->m_fixed)
    {
        log()->get(LogLevel::Debug2) << "Calculating ICP\n";
        PointViewPtr result = this->icp(this->m_fixed, view);
        viewSet.insert(result);
        log()->get(LogLevel::Debug2) << "ICP complete\n";
        this->m_complete = true;
    }
    else
    {
        log()->get(LogLevel::Debug2) << "Adding fixed points\n";
        this->m_fixed = view;
    }
    return viewSet;
}

void IterativeClosestPoint::done(PointTableRef _)
{
    if (!this->m_complete)
    {
        throw pdal_error(
            "filters.icp must have two point view inputs, no more, no less");
    }
}

PointViewPtr IterativeClosestPoint::icp(PointViewPtr fixed,
                                        PointViewPtr moving) const
{
    // rather, compute centroid and simply overwrite minx|y|z
    std::vector<PointId> ids(fixed->size());
    std::iota(ids.begin(), ids.end(), 0);
    auto centroid = eigen::computeCentroid(*fixed, ids);

    // initialize the "final_transformation" to Identity, though we could allow
    // the user to provide an initial guess
    Eigen::Matrix4d final_transformation = Eigen::Matrix4d::Identity();

    // demean the fixed dataset once and only once
    PointViewPtr tempFixed = fixed->demeanPointView();

    // construct KDtree of the fixed dataset so that the moving one can
    // search for the nearest point
    KD3Index& kd_fixed = tempFixed->build3dIndex();

    // iterate to max number of iterations or until converged (hardcoded for
    // now)
    bool converged(false);
    for (int iter = 0; iter < m_max_iters; ++iter)
    {
        // apply the previously computed transformation to demeaned moving
        // dataset
        PointViewPtr tempMoving = moving->makeNew();
        for (PointId i = 0; i < moving->size(); ++i)
        {
            double x =
                moving->getFieldAs<double>(Dimension::Id::X, i) - centroid.x();
            double y =
                moving->getFieldAs<double>(Dimension::Id::Y, i) - centroid.y();
            double z =
                moving->getFieldAs<double>(Dimension::Id::Z, i) - centroid.z();
            tempMoving->setField(Dimension::Id::X, i,
                                 x * final_transformation.coeff(0, 0) +
                                     y * final_transformation.coeff(0, 1) +
                                     z * final_transformation.coeff(0, 2) +
                                     final_transformation.coeff(0, 3));
            tempMoving->setField(Dimension::Id::Y, i,
                                 x * final_transformation.coeff(1, 0) +
                                     y * final_transformation.coeff(1, 1) +
                                     z * final_transformation.coeff(1, 2) +
                                     final_transformation.coeff(1, 3));
            tempMoving->setField(Dimension::Id::Z, i,
                                 x * final_transformation.coeff(2, 0) +
                                     y * final_transformation.coeff(2, 1) +
                                     z * final_transformation.coeff(2, 2) +
                                     final_transformation.coeff(2, 3));
        }

        // find correspondences and compute mean square error
        std::vector<PointId> fixed_idx;
        std::vector<PointId> moving_idx;
        double mse(0.0);
        for (PointId i = 0; i < tempMoving->size(); ++i)
        {
            // find nearest neighbor
            PointRef p = tempMoving->point(i);
            std::vector<PointId> indices(1);
            std::vector<double> sqr_dists(1);
            kd_fixed.knnSearch(p, 1, &indices, &sqr_dists);

            // check that dist is less than thresh
            // just be careful to check the sqrt if I allow the user to set this
            // manually, in fact this is all moot until I do.
            // double max_dist = std::numeric_limits<double>::max();
            // if (sqr_dists[0] > max_dist)
            //    continue;

            // store indices of correspondences along with distance
            moving_idx.push_back(i);
            fixed_idx.push_back(indices[0]);
            mse += std::sqrt(sqr_dists[0]);
        }
        mse /= fixed_idx.size();
        log()->get(LogLevel::Debug2) << "MSE: " << mse << std::endl;

        // estimate transformation

        // demean fixed dataset, subtract same offsets from moving, then
        // estimate rigid transformation using Umeyama method (same thing PCL
        // did)
        auto A = eigen::pointViewToEigen(*tempFixed, fixed_idx);
        auto B = eigen::pointViewToEigen(*tempMoving, moving_idx);
        auto T = Eigen::umeyama(B.transpose(), A.transpose(), false);
        log()->get(LogLevel::Debug2) << "Current dx: " << T.coeff(0, 3) << ", "
                                     << "dy: " << T.coeff(1, 3) << std::endl;

        // 2. The epsilon (difference) between the previous transformation and
        // the current estimated transformation
        double cos_angle =
            0.5 * (T.coeff(0, 0) + T.coeff(1, 1) + T.coeff(2, 2) - 1);
        double translation_sqr = T.coeff(0, 3) * T.coeff(0, 3) +
                                 T.coeff(1, 3) * T.coeff(1, 3) +
                                 T.coeff(2, 3) * T.coeff(2, 3);
        log()->get(LogLevel::Debug2) << "Rotation: " << cos_angle << std::endl;
        log()->get(LogLevel::Debug2)
            << "Translation: " << translation_sqr << std::endl;

        double rotation_threshold(0.99999);        // 0.256 degrees
        double translation_threshold(3e-4 * 3e-4); // 0.0003 meters
        double mse_threshold_relative(
            0.00001); // 0.001% of the previous MSE (relative error)
        double mse_threshold_absolute(1e-12); // MSE (absolute error)

        final_transformation = final_transformation * T;
        log()->get(LogLevel::Debug2)
            << "Cumulative dx: " << final_transformation.coeff(0, 3) << ", "
            << "dy: " << final_transformation.coeff(1, 3) << std::endl;

        if ((cos_angle >= rotation_threshold) &&
            (translation_sqr <= translation_threshold))
        {
            converged = true;
            log()->get(LogLevel::Debug2) << "converged\n";
            break;
        }

        /*

            // 3. The relative sum of Euclidean squared errors is smaller than a
           user defined threshold
            // Absolute
            if (fabs (mse - correspondences_prev_mse_) < mse_threshold_absolute)
            {
                if (iterations_similar_transforms_ >=
           max_iterations_similar_transforms_)
                {
                    convergence_state_ = CONVERGENCE_CRITERIA_ABS_MSE;
                    return (true);
                }
                is_similar = true;
            }

            // Relative
            if (fabs (mse - correspondences_prev_mse_) /
           correspondences_prev_mse_ < mse_threshold_relative)
            {
                if (iterations_similar_transforms_ >=
           max_iterations_similar_transforms_)
                {
                    convergence_state_ = CONVERGENCE_CRITERIA_REL_MSE;
                    return (true);
                }
                is_similar = true;
            }
        */
    }

    // apply the previously computed transformation
    for (PointId i = 0; i < moving->size(); ++i)
    {
        double x =
            moving->getFieldAs<double>(Dimension::Id::X, i) - centroid.x();
        double y =
            moving->getFieldAs<double>(Dimension::Id::Y, i) - centroid.y();
        double z =
            moving->getFieldAs<double>(Dimension::Id::Z, i) - centroid.z();
        moving->setField(Dimension::Id::X, i,
                         x * final_transformation.coeff(0, 0) +
                             y * final_transformation.coeff(0, 1) +
                             z * final_transformation.coeff(0, 2) +
                             final_transformation.coeff(0, 3) + centroid.x());
        moving->setField(Dimension::Id::Y, i,
                         x * final_transformation.coeff(1, 0) +
                             y * final_transformation.coeff(1, 1) +
                             z * final_transformation.coeff(1, 2) +
                             final_transformation.coeff(1, 3) + centroid.y());
        moving->setField(Dimension::Id::Z, i,
                         x * final_transformation.coeff(2, 0) +
                             y * final_transformation.coeff(2, 1) +
                             z * final_transformation.coeff(2, 2) +
                             final_transformation.coeff(2, 3) + centroid.z());
    }

    // obtain final MSE fitness
    double mse(0.0);
    KD3Index& kd_fixed_orig = fixed->build3dIndex();
    for (PointId i = 0; i < moving->size(); ++i)
    {
        // find nearest neighbor
        PointRef p = moving->point(i);
        std::vector<PointId> indices(1);
        std::vector<double> sqr_dists(1);
        kd_fixed_orig.knnSearch(p, 1, &indices, &sqr_dists);
        mse += std::sqrt(sqr_dists[0]);
    }
    mse /= moving->size();
    log()->get(LogLevel::Debug2) << "MSE: " << mse << std::endl;

    // populate metadata nodes
    MetadataNode root = getMetadata();
    root.add("transform",
        Eigen::MatrixXd(final_transformation.cast<double>()));
    root.add("converged", converged);
    root.add("fitness", mse);

    return moving;
}

} // namespace pdal
